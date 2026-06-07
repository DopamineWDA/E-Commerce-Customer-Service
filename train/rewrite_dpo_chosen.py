#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Rewrite DPO chosen replies with qwen-plus while keeping rejected unchanged.

The script is intentionally JSONL-in/JSONL-out and resumable. It writes one
cache record per rewritten pair, then materializes a full TRL-compatible DPO
file with the original prompt/rejected/reference/metadata preserved.
"""

from __future__ import annotations

import argparse
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


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from evaluate_inference_judge import JUDGE_SYSTEM_PROMPT  # noqa: E402


REWRITE_SYSTEM_PROMPT = f"""
你是电商客服 DPO chosen 改写专家。你的任务是把原 chosen 改写成更符合高分客服 judge 的回复，
只改写 chosen，不改 rejected，不改变用户问题的业务结论。

下面是当前 judge 的高分标准，请严格对齐：
{JUDGE_SYSTEM_PROMPT}

改写硬性规则：
1. 必须保留原 chosen 的核心业务结论，不改变“支持/不支持/部分支持/可退/不可退/需确认/以页面为准”等结论。
   即使参考答案、rejected 或 ranking 诊断与原 chosen 结论冲突，也不得借改写纠正原 chosen 的业务结论。
2. 不新增具体订单状态、物流节点、金额、库存、日期时间、后台已完成操作、已加急、已备注、已退款、已补发等无依据事实。
   不新增原 chosen 未提到的费用承担、赔付、补贴、赠券、上门取件、免费维修、运费返还等具体政策承诺。
3. 强烈禁止新增以下内容，即使为了让答案更完整也不允许：
   - 订单状态，例如“已发货”“已出库”“待审核”“当前显示”
   - 物流节点，例如“已揽收”“发往中转站”“派送中”“已签收”
   - 发货时间、到货时间、预计具体日期、几点前送达
   - 后台审核、自动处理、系统同步、系统自动触发、系统已匹配
   - 仓库动作、打包、出库、今晚发出、优先发货
   - 补发机制、优先通道、直连物流、异常预警
   - 已完成操作，例如“已为您处理/备注/加急/申请/退款/拦截”
   - 原 chosen 未提到的平台规则细节，例如费用承担、赔付、补贴、赠券、上门取件、免费维修、运费返还
4. 允许增强且优先增强：
   - 条件化表达
   - 用户下一步动作
   - 页面操作路径
   - 风险提示
   - 联系客服时机
   - 条件判断
   例如：“呼伦贝尔属于偏远地区，物流时效可能会略有延长，建议以订单页预计送达时间为准；若长时间未更新，可联系客服进一步核实。”
   不要写成：“系统已自动匹配优先物流线路，若超时将自动触发异常预警。”
5. 如果需要表达不确定信息，使用“可能、建议、以页面显示为准、可联系客服核实”，不要写成系统事实或后台承诺。
6. 优先补充操作路径、下一步动作、条件判断，使用户知道接下来怎么做。
7. 避免空话，例如“请耐心等待”“我们会尽力处理”“请联系客服处理”这类没有路径或动作的泛泛表达。
8. 不强制长度比例；通常控制在 1 到 3 句，能补清楚路径和条件即可。宁可简洁，不要写成长段说明。
9. 输出像真实客服，不要像规则说明，不要列评分维度，不要解释你如何改写。
10. 不要输出思考过程、Markdown、编号列表或 JSON 以外的文本。

只输出严格 JSON：
{{
  "rewritten_chosen": "改写后的客服回复",
  "rationale": "一句话说明补强点"
}}
""".strip()


RISKY_POLICY_TERMS = [
    "已发货",
    "已出库",
    "待审核",
    "当前显示",
    "已揽收",
    "中转站",
    "派送中",
    "已签收",
    "今天",
    "明天",
    "今晚",
    "小时内",
    "后台",
    "审核",
    "自动",
    "系统同步",
    "系统已",
    "系统将",
    "系统会",
    "触发",
    "异常预警",
    "仓库",
    "打包",
    "出库",
    "发出",
    "优先",
    "通道",
    "直连",
    "拦截",
    "已为您",
    "已帮您",
    "帮您处理",
    "帮您备注",
    "帮您加急",
    "已处理",
    "已备注",
    "已加急",
    "已退款",
    "已补发",
    "运费",
    "费用",
    "免费",
    "赔付",
    "赔偿",
    "补偿",
    "补发",
    "返还",
    "退还",
    "垫付",
    "上门取件",
    "取件",
    "赠券",
    "优惠券",
    "维修",
]


RISKY_ADDED_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"(预计|最晚|大概|约).{0,8}(\d+\s*[-~到]\s*\d+\s*(天|个工作日|小时)|\d+\s*(天|个工作日|小时)内)",
        r"(预计|最晚|大概|约).{0,12}(今天|明天|今晚|本周|下周|周[一二三四五六日天]|星期[一二三四五六日天])",
        r"\d{1,2}[:：]\d{2}",
        r"\d{1,2}\s*点",
        r"\d{1,2}\s*月\s*\d{1,2}\s*日",
        r"\d{1,2}\s*[-/]\s*\d{1,2}",
    ]
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite DPO chosen replies with qwen-plus.")
    parser.add_argument("--input", type=str, default="/home/txs/work/zyp/SFT_DPO/data_dpo/dpo_pairs.jsonl")
    parser.add_argument("--output", type=str, default="/home/txs/work/zyp/SFT_DPO/data_dpo/dpo_pairs_rewritten.jsonl")
    parser.add_argument("--cache-output", type=str, default="/home/txs/work/zyp/SFT_DPO/data_dpo/rewrite_chosen_cache.jsonl")
    parser.add_argument("--failed-output", type=str, default="/home/txs/work/zyp/SFT_DPO/data_dpo/rewrite_chosen_failed.jsonl")
    parser.add_argument("--summary-output", type=str, default="/home/txs/work/zyp/SFT_DPO/data_dpo/rewrite_chosen_summary.json")
    parser.add_argument("--model", type=str, default="qwen-plus")
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
        help="Environment variable containing the API key.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--retry-max-sleep", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_cache(path: Path, overwrite: bool) -> dict[str, dict[str, Any]]:
    if overwrite or not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") and record.get("rewritten_chosen"):
                cache[str(record["id"])] = record
    return cache


def format_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for idx, message in enumerate(messages, start=1):
        lines.append(f"{idx}. {message.get('role', '')}: {message.get('content', '')}")
    return "\n".join(lines)


def build_rewrite_prompt(record: dict[str, Any]) -> str:
    prompt = record.get("prompt") or []
    chosen = (record.get("chosen") or [{}])[0].get("content", "")
    rejected = (record.get("rejected") or [{}])[0].get("content", "")
    reference = str(record.get("reference", "")).strip()
    metadata = record.get("metadata") or {}
    source_metadata = metadata.get("source_metadata") or {}
    ranking = metadata.get("ranking") or {}
    original_length = len(chosen)
    max_target_length = min(220, max(original_length + 80, int(original_length * 2.2)))

    return f"""请改写下面 DPO pair 的 chosen。只输出 JSON。

[样本ID]
{record.get("id", "")}

[对话上下文]
{format_messages(prompt)}

[参考答案，仅用于理解背景；不得用它推翻或纠正原 chosen 的业务结论]
{reference}

[原 chosen：需要改写，必须保留业务结论；不得把“支持/部分支持/不支持/需确认”等立场改掉]
{chosen}

[长度控制]
原 chosen 长度约 {original_length} 个中文字符；不强制按比例扩写，建议不超过约 {max_target_length} 个中文字符。宁可简洁，也不要为了补充路径而加入无依据政策细节。

[rejected：保持不变；改写后的 chosen 应明显比它更准确、更闭环、更可执行]
{rejected}

[原 ranking 诊断，可参考但不要复述]
question_type={source_metadata.get("question_type")}
stance={source_metadata.get("stance")}
score_delta={metadata.get("score_delta")}
rationale={ranking.get("rationale", "")}

请输出：
{{
  "rewritten_chosen": "...",
  "rationale": "..."
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


def normalize_rewrite(raw: dict[str, Any], original_chosen: str) -> dict[str, str]:
    rewritten = str(raw.get("rewritten_chosen", "")).strip()
    rewritten = re.sub(r"(?s)<think>.*?</think>", "", rewritten).strip()
    rationale = str(raw.get("rationale", "")).strip()
    if not rewritten:
        raise ValueError("rewritten_chosen is empty")
    if rewritten == original_chosen.strip():
        raise ValueError("rewritten_chosen is unchanged")
    if len(rewritten) < 8:
        raise ValueError("rewritten_chosen is too short")
    added_risky_terms = [
        term
        for term in RISKY_POLICY_TERMS
        if term in rewritten and term not in original_chosen
    ]
    if added_risky_terms:
        raise ValueError(f"rewritten_chosen adds risky policy terms absent from original chosen: {added_risky_terms}")
    added_risky_patterns = [
        pattern.pattern
        for pattern in RISKY_ADDED_PATTERNS
        if pattern.search(rewritten) and not pattern.search(original_chosen)
    ]
    if added_risky_patterns:
        raise ValueError(f"rewritten_chosen adds risky time/status pattern absent from original chosen: {added_risky_patterns}")
    return {"rewritten_chosen": rewritten, "rationale": rationale}


def post_chat_completion(body: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"API key is not set. Export {args.api_key_env} before running this script.")
    if not args.api_base:
        raise RuntimeError("API base URL is not set. Export OPENAI_BASE_URL or pass --api-base.")

    request = urllib.request.Request(
        args.api_base.rstrip("/") + "/chat/completions",
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
        except Exception:
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


def rewrite_one(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    original_chosen = (record.get("chosen") or [{}])[0].get("content", "")
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": build_rewrite_prompt(record)},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    last_error = ""
    for attempt in range(max(1, args.max_retries)):
        try:
            payload = post_chat_completion(body, args)
            content = payload["choices"][0]["message"]["content"]
            rewrite = normalize_rewrite(parse_json_content(content), original_chosen)
            return {
                "id": record["id"],
                "model": args.model,
                "original_chosen": original_chosen,
                **rewrite,
                "original_length": len(original_chosen),
                "rewritten_length": len(rewrite["rewritten_chosen"]),
            }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
        ) as exc:
            last_error = format_exception(exc)
            if attempt + 1 < max(1, args.max_retries):
                sleep_before_retry(attempt, exc, args)
    raise RuntimeError(f"rewrite failed for id={record.get('id')}: {last_error}")


def apply_rewrites(records: list[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rewritten_records = []
    for record in records:
        item = json.loads(json.dumps(record, ensure_ascii=False))
        cached = cache.get(str(item.get("id")))
        if cached and cached.get("rewritten_chosen"):
            original = item["chosen"][0]["content"]
            item["chosen"][0]["content"] = cached["rewritten_chosen"]
            item.setdefault("metadata", {})["chosen_rewrite"] = {
                "model": cached.get("model"),
                "original_chosen": original,
                "rationale": cached.get("rationale", ""),
                "original_length": cached.get("original_length"),
                "rewritten_length": cached.get("rewritten_length"),
            }
        rewritten_records.append(item)
    return rewritten_records


def summarize(records: list[dict[str, Any]], cache: dict[str, dict[str, Any]], failed: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = []
    for item in cache.values():
        original_length = item.get("original_length") or 0
        rewritten_length = item.get("rewritten_length") or 0
        if original_length:
            ratios.append(rewritten_length / original_length)
    ratios_sorted = sorted(ratios)
    def pct(q: float) -> float | None:
        if not ratios_sorted:
            return None
        index = min(len(ratios_sorted) - 1, max(0, int(round((len(ratios_sorted) - 1) * q))))
        return round(ratios_sorted[index], 4)

    return {
        "input_records": len(records),
        "rewritten_records": len(cache),
        "failed_records": len(failed),
        "length_ratio": {
            "min": pct(0.0),
            "p10": pct(0.1),
            "median": pct(0.5),
            "p90": pct(0.9),
            "max": pct(1.0),
            "mean": round(sum(ratios) / len(ratios), 4) if ratios else None,
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    cache_path = Path(args.cache_output).resolve()
    failed_path = Path(args.failed_output).resolve()
    summary_path = Path(args.summary_output).resolve()

    records = read_jsonl(input_path)
    if args.limit is not None:
        records = records[: max(0, args.limit)]

    if args.overwrite:
        for path in [cache_path, failed_path, output_path, summary_path]:
            if path.exists():
                path.unlink()

    cache = load_existing_cache(cache_path, overwrite=False)
    pending = [record for record in records if str(record.get("id")) not in cache]
    failed: list[dict[str, Any]] = []

    if pending:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
            future_to_record = {executor.submit(rewrite_one, record, args): record for record in pending}
            batch: list[dict[str, Any]] = []
            failed_batch: list[dict[str, Any]] = []
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    rewritten = future.result()
                    cache[str(record["id"])] = rewritten
                    batch.append(rewritten)
                except Exception as exc:
                    failed_record = {
                        "id": record.get("id"),
                        "error": format_exception(exc),
                        "original_chosen": (record.get("chosen") or [{}])[0].get("content", ""),
                    }
                    failed.append(failed_record)
                    failed_batch.append(failed_record)

                done += 1
                if batch:
                    append_jsonl(cache_path, batch)
                    batch = []
                if failed_batch:
                    append_jsonl(failed_path, failed_batch)
                    failed_batch = []
                if done % max(1, args.progress_every) == 0 or done == len(pending):
                    print(f"rewrite progress: {done}/{len(pending)} newly processed, total cached={len(cache)}, failed={len(failed)}", flush=True)

    rewritten_records = apply_rewrites(records, cache)
    write_jsonl(output_path, rewritten_records)
    summary = summarize(records, cache, failed)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
