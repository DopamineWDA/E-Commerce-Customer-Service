#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Evaluate SFT inference samples with a Qwen/OpenAI-compatible judge.

Input:
  inference_samples.json
    [
      {
        "prompt_messages": [...],
        "reference": "...",
        "prediction": "..."
      }
    ]

Output:
  judge_results.jsonl
    one judged record per sample

  judge_summary.json / judge_summary.csv
    aggregate metrics:
      Accuracy = mean(accuracy)
      Auto Resolve = mean(auto_resolve)
      Closure Rate = mean(closure)
      CSAT = mean(satisfaction)

The script uses the OpenAI-compatible chat-completions API exposed by
DashScope/Qwen-compatible services. It intentionally avoids SDK dependencies.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


JUDGE_SYSTEM_PROMPT = """
你是电商客服质检评审员。你的任务是评估模型最后一条客服回复是否适合真实电商客服场景。

请根据：

* 对话上下文
* 参考答案
* 模型回复
* 常识性电商客服规范

综合评分。

不要因为模型措辞与参考答案不完全一致而扣分；
重点评估：

* 是否合理解决用户问题
* 是否提供下一步动作
* 是否符合真实客服体验

允许：

* 合理的行业常见规则推断
* 常见物流时效表达
* 常见平台操作建议

但禁止：

* 明显虚构订单状态
* 明显虚构系统操作已完成
* 明显违背上下文或常识

四个评分字段：

1. accuracy: 0或1
   回复是否准确回答了用户最后的问题，
   且没有答非所问、
   明显虚构平台状态、
   或明显错误政策。

允许一定合理业务推断。

2. auto_resolve: 0或1
   回复是否已经提供：

* 明确下一步动作
* 操作路径
* 或用户可继续执行的方案

即使用户后续还有问题，
只要用户已经知道下一步怎么做，
即可判为1。

3. closure: 0或1
   回复是否形成闭环：

* 给出明确结论
* 给出下一步建议
* 给出操作方向
* 给出条件判断

看是否用户在回复后已经有明确的结论结果不需要再进行提问了。

4. satisfaction: 1~5整数

5 = 准确、自然、可执行、像真实客服
4 = 基本正确，有帮助，略泛
3 = 部分有用，但信息不足
2 = 明显含糊或存在风险
1 = 严重答非所问、明显违规或严重虚构

必须输出严格JSON：
{
"accuracy":0,
"auto_resolve":0,
"closure":0,
"satisfaction":1
}


"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge SFT inference samples with qwen-plus.")
    parser.add_argument(
        "--input",             
        type=str,
        required=True,
        help="Path to inference_samples.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for judge_results.jsonl and judge_summary files. Defaults to input file directory.",
    )
    parser.add_argument("--results-output", type=str, default=None, help="Override per-sample JSONL output path.")
    parser.add_argument("--failed-output", type=str, default=None, help="Override failed-sample JSONL output path.")
    parser.add_argument("--summary-output", type=str, default=None, help="Override summary JSON output path.")
    parser.add_argument("--csv-output", type=str, default=None, help="Override summary CSV output path.")
    parser.add_argument("--model", type=str, default="qwen-plus", help="Judge model.")
    parser.add_argument(
        "--api-base",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible API base URL. Defaults to the OPENAI_BASE_URL environment variable.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge sampling temperature.")
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count per sample.")
    parser.add_argument("--retry-base-sleep", type=float, default=2.0, help="Base exponential-backoff sleep in seconds.")
    parser.add_argument("--retry-max-sleep", type=float, default=30.0, help="Max sleep between retries in seconds.")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent judge requests.")
    parser.add_argument("--progress-every", type=int, default=1, help="Print progress after every N newly judged samples.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any sample fails after retries.")
    parser.add_argument("--overwrite", action="store_true", help="Ignore existing judge_results.jsonl and rewrite.")
    return parser.parse_args()


def load_samples(path: Path, limit: int | None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file_obj:
        samples = json.load(file_obj)
    if not isinstance(samples, list):
        raise ValueError("inference samples must be a JSON list")
    if limit is not None:
        samples = samples[: max(0, limit)]
    return samples


def default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_path.parent
    results_path = Path(args.results_output).resolve() if args.results_output else output_dir / "judge_results.jsonl"
    failed_path = Path(args.failed_output).resolve() if args.failed_output else output_dir / "judge_failed.jsonl"
    summary_path = Path(args.summary_output).resolve() if args.summary_output else output_dir / "judge_summary.json"
    csv_path = Path(args.csv_output).resolve() if args.csv_output else output_dir / "judge_summary.csv"
    return results_path, failed_path, summary_path, csv_path


def load_existing_results(path: Path, overwrite: bool) -> dict[int, dict[str, Any]]:
    if overwrite or not path.exists():
        return {}

    existing: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_index = int(record["sample_index"])
            existing[sample_index] = record
    return existing


def format_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for idx, message in enumerate(messages):
        role = message.get("role", "")
        content = message.get("content", "")
        lines.append(f"{idx + 1}. {role}: {content}")
    return "\n".join(lines)


def build_judge_prompt(sample_index: int, sample: dict[str, Any]) -> str:
    prompt_messages = sample.get("prompt_messages") or []
    reference = str(sample.get("reference", "")).strip()
    prediction = str(sample.get("prediction", "")).strip()
    return f"""请评估第 {sample_index} 条样本中模型最后一条客服回复。

[对话上下文]
{format_messages(prompt_messages)}

[参考答案]
{reference}

[模型最后回复]
{prediction}

请输出如下 JSON 格式：
{{
  "accuracy": 0或1,
  "auto_resolve": 0或1,
  "closure": 0或1,
  "satisfaction": 1到5的整数,
  "rationale": "一句话说明主要原因"
}}"""


def parse_json_content(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_binary(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in {0, 1}:
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise ValueError(f"{field_name} must be 0 or 1, got {value!r}")


def normalize_satisfaction(value: Any) -> int:
    if isinstance(value, (int, float)) and 1 <= int(value) <= 5:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
        if 1 <= number <= 5:
            return number
    raise ValueError(f"satisfaction must be an integer from 1 to 5, got {value!r}")


def normalize_judgement(raw: dict[str, Any]) -> dict[str, Any]:
    judgement = {
        "accuracy": normalize_binary(raw.get("accuracy"), "accuracy"),
        "auto_resolve": normalize_binary(raw.get("auto_resolve"), "auto_resolve"),
        "closure": normalize_binary(raw.get("closure"), "closure"),
        "satisfaction": normalize_satisfaction(raw.get("satisfaction")),
        "rationale": str(raw.get("rationale", "")).strip(),
    }
    return judgement


def post_chat_completion(body: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"API key is not set. Please export {args.api_key_env} before running this script.")
    if not args.api_base:
        raise RuntimeError("API base URL is not set. Export OPENAI_BASE_URL or pass --api-base.")

    url = args.api_base.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def format_exception(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
        except Exception:  # noqa: BLE001
            detail = ""
        return f"HTTP Error {exc.code}: {exc.reason}; body={detail}"
    return f"{type(exc).__name__}: {exc}"


def sleep_before_retry(attempt: int, exc: Exception, args: argparse.Namespace) -> None:
    retry_after = None
    if isinstance(exc, urllib.error.HTTPError):
        retry_after_header = exc.headers.get("Retry-After")
        if retry_after_header:
            try:
                retry_after = float(retry_after_header)
            except ValueError:
                retry_after = None

    if retry_after is None:
        retry_after = min(args.retry_max_sleep, args.retry_base_sleep * (2**attempt))
        retry_after += random.uniform(0, min(1.0, args.retry_base_sleep))
    time.sleep(max(0.0, retry_after))


def judge_one(sample_index: int, sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": build_judge_prompt(sample_index, sample)},
    ]
    body = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
    }

    last_error = ""
    for attempt in range(max(1, args.max_retries)):
        try:
            payload = post_chat_completion(body, args)
            content = payload["choices"][0]["message"]["content"]
            judgement = normalize_judgement(parse_json_content(content))
            return {
                "sample_index": sample_index,
                "prompt_messages": sample.get("prompt_messages", []),
                "reference": sample.get("reference", ""),
                "prediction": sample.get("prediction", ""),
                "judge_model": args.model,
                **judgement,
            }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = format_exception(exc)
            if attempt + 1 < max(1, args.max_retries):
                sleep_before_retry(attempt, exc, args)

    raise RuntimeError(f"sample {sample_index} judge failed: {last_error}")


def make_failed_record(sample_index: int, sample: dict[str, Any], error: Exception, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "sample_index": sample_index,
        "prompt_messages": sample.get("prompt_messages", []),
        "reference": sample.get("reference", ""),
        "prediction": sample.get("prediction", ""),
        "judge_model": args.model,
        "error": format_exception(error),
    }


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def rewrite_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_summary(records: list[dict[str, Any]], input_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        raise ValueError("no judged records to summarize")

    def mean(field_name: str) -> float:
        return sum(float(record[field_name]) for record in records) / count

    return {
        "input": str(input_path),
        "judge_model": args.model,
        "num_samples": count,
        "accuracy": round(mean("accuracy"), 6),
        "auto_resolve": round(mean("auto_resolve"), 6),
        "closure_rate": round(mean("closure"), 6),
        "csat": round(mean("satisfaction"), 6),
        "accuracy_percent": round(mean("accuracy") * 100, 2),
        "auto_resolve_percent": round(mean("auto_resolve") * 100, 2),
        "closure_rate_percent": round(mean("closure") * 100, 2),
        "csat_1_to_5": round(mean("satisfaction"), 3),
    }


def save_summary(summary: dict[str, Any], summary_path: Path, csv_path: Path) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
        fieldnames = [
            "input",
            "judge_model",
            "num_samples",
            "accuracy",
            "auto_resolve",
            "closure_rate",
            "csat",
            "accuracy_percent",
            "auto_resolve_percent",
            "closure_rate_percent",
            "csat_1_to_5",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({key: summary.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    results_path, failed_path, summary_path, csv_path = default_paths(args)

    samples = load_samples(input_path, args.limit)
    existing = load_existing_results(results_path, args.overwrite)
    pending = [(idx, sample) for idx, sample in enumerate(samples) if idx not in existing]

    if args.overwrite:
        for path in (results_path, failed_path):
            if path.exists():
                path.unlink()

    print(
        json.dumps(
            {
                "input": str(input_path),
                "samples": len(samples),
                "already_judged": len(existing),
                "pending": len(pending),
                "results_output": str(results_path),
                "failed_output": str(failed_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    newly_completed: list[dict[str, Any]] = []
    newly_failed: list[dict[str, Any]] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            future_to_index = {
                executor.submit(judge_one, sample_index, sample, args): sample_index
                for sample_index, sample in pending
            }
            for future in as_completed(future_to_index):
                sample_index = future_to_index[future]
                sample = samples[sample_index]
                try:
                    record = future.result()
                    newly_completed.append(record)
                    append_jsonl(results_path, [record])
                except Exception as exc:  # noqa: BLE001
                    failed_record = make_failed_record(sample_index, sample, exc, args)
                    newly_failed.append(failed_record)
                    append_jsonl(failed_path, [failed_record])
                    print(
                        f"Judge failed sample_index={sample_index}: {failed_record['error']}",
                        file=sys.stderr,
                        flush=True,
                    )

                progress_every = max(1, args.progress_every)
                finished = len(newly_completed) + len(newly_failed)
                if finished % progress_every == 0 or finished == len(pending):
                    total_completed = len(existing) + len(newly_completed)
                    print(
                        f"Judge progress: {finished}/{len(pending)} attempted, "
                        f"succeeded={len(newly_completed)}, failed={len(newly_failed)}, "
                        f"{total_completed}/{len(samples)} total succeeded",
                        flush=True,
                    )
    except Exception as exc:  # noqa: BLE001
        print(f"Judge stopped: {exc}", file=sys.stderr, flush=True)
        if args.fail_on_error:
            raise SystemExit(1) from exc

    all_records = list(existing.values()) + newly_completed
    all_records = sorted(all_records, key=lambda item: int(item["sample_index"]))
    if all_records:
        rewrite_jsonl(results_path, all_records)
        summary = compute_summary(all_records, input_path, args)
        save_summary(summary, summary_path, csv_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if newly_failed and args.fail_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
