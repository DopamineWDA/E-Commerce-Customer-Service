#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Build conversational DPO preference data with SFT multi-sampling + qwen-plus ranking.

Flow:
1. Randomly sample rows from SFT train.jsonl.
2. Keep system prompt, dialogue history, and the last user turn as the prompt.
   The original final assistant reply is removed from the generation input.
3. Generate 2-3 candidate replies from the same SFT checkpoint with different
   temperature/top-p settings.
4. Ask qwen-plus to score and rank the candidates. qwen-plus only judges; it
   does not generate the chosen answer.
5. Write TRL-compatible conversational DPO JSONL:
   {"prompt": [...], "chosen": [...], "rejected": [...]}.
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

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "trl_qwen_customer_sft.yaml"
DEFAULT_TRAIN_FILE = Path("/home/txs/work/zyp/SFT_DPO/data_clustered_5.7_1/train.jsonl")
DEFAULT_ADAPTER_PATH = Path("/home/txs/work/zyp/SFT_DPO/train/outputs/sft_ablation/rank16_qvko/checkpoint")
DEFAULT_OUTPUT_DIR = Path("/home/txs/work/zyp/SFT_DPO/data_dpo")


RANK_SYSTEM_PROMPT = """你是严格的电商客服偏好标注员。你的任务是对同一对话上下文下的多个候选客服回复进行质检、评分和排序。

请只根据：
* 对话上下文
* 候选客服回复
* 常识性电商客服规范

综合判断。不要依赖参考答案，也不要要求候选与某个参考措辞一致。

评估维度必须贴近最终客服judge：
1. hallucination: 0或1
   1=严重幻觉，只包括明显虚构订单号、明显虚构订单状态、虚构后台已操作、伪造物流节点、伪造金额/时间/库存、或编造不存在的平台能力/政策。
   0=没有严重幻觉。合理业务推断、条件化建议、常见平台流程说明、常见售后规则、常见物流时效表达不应视为hallucination。
   只有严重 hallucination=1 的候选绝对不能作为chosen。
2. accuracy: 0或1
   是否准确回答用户最后的问题，且没有答非所问、明显虚构平台状态或明显错误政策。
3. closure: 0或1
   是否形成闭环：给出明确结论、下一步建议、操作方向或条件判断，使用户知道接下来怎么做。
4. auto_resolve: 0或1
   是否提供明确下一步动作、操作路径或用户可继续执行的方案。
5. satisfaction: 1~5整数
   5=准确、自然、可执行、像真实优秀客服
   4=基本正确，有帮助，略泛
   3=部分有用，但信息不足
   2=明显含糊或存在风险
   1=严重答非所问、明显违规或严重虚构
6. overall_score: 0~100数字
   用于表达偏好强度。必须让更贴近高分judge的候选分数更高，并尽量拉开好坏候选差距。

必须避免过度保守：
- 不要把常见业务推断、条件化建议、平台常见路径、售后常见规则误判为幻觉。
- 优秀客服应能在不编造后台事实的前提下给出明确路径，例如“我的订单→申请售后/申请退款”“若未发货可取消或申请退款”“若已发货可拒收或收到后申请售后”。
- 明确惩罚“强闭环假客服”：例如“已为您备注加急”“物流已同步”“仓库今晚发出”“我已帮您处理完成”等没有上下文依据的后台动作或状态。

排序原则：
- chosen 应偏向 grounded、条件化、有操作路径、有下一步、有结论、有真实客服感、且不编造后台状态的执行型回复。
- 程序只会把严重 hallucination=1 的候选排除在chosen之外；其余候选按 accuracy、closure、auto_resolve、satisfaction、overall_score 选择chosen。
- rejected 应优先选择“基本正确但不够具体、安全但无帮助、不闭环、只有安抚、缺少操作路径”的候选，而不是总选择明显错误或严重hallucination候选。
- 典型优质pair：chosen=“您可以在【我的订单→申请退款】提交申请，若订单尚未发货，一般会原路退款。”；rejected=“请联系客服处理。”
- 如果多个候选接近，也要根据可执行性、闭环性、条件判断、真实客服感拉开 overall_score，避免所有分数完全一样。

只输出严格 JSON，不要输出解释性正文：
{
  "scores": [
    {
      "index": 0,
      "hallucination": 0,
      "accuracy": 0,
      "closure": 0,
      "auto_resolve": 0,
      "satisfaction": 1,
      "overall_score": 1.0,
      "rationale": "简短原因"
    }
  ],
  "rationale": "整体排序依据的简短说明"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO pairs using SFT candidate sampling and qwen-plus ranking.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="YAML config with system prompt.")
    parser.add_argument("--train-file", type=str, default=str(DEFAULT_TRAIN_FILE), help="SFT train JSONL.")
    parser.add_argument("--adapter-path", type=str, default=str(DEFAULT_ADAPTER_PATH), help="LoRA SFT adapter checkpoint.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--sample-size", type=int, default=3000, help="Number of train samples to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random sampling seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite all intermediate outputs.")

    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.5,0.7,0.9",
        help="Comma-separated temperatures for SFT candidate sampling.",
    )
    parser.add_argument(
        "--top-ps",
        type=str,
        default="0.9,0.9,0.9",
        help="Comma-separated top-p values. If one value is given, it is reused for all temperatures.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Max new tokens for each SFT candidate.")
    parser.add_argument("--sft-batch-size", type=int, default=4, help="Batch size for local SFT generation.")
    parser.add_argument("--skip-sft", action="store_true", help="Reuse sft_candidates.jsonl.")
    parser.add_argument("--skip-ranking", action="store_true", help="Reuse qwen_ranked.jsonl.")
    parser.add_argument(
        "--discard-fraction",
        type=float,
        default=0.10,
        help="Drop this fraction of pairs with the smallest best_score - worst_score.",
    )

    parser.add_argument("--rank-model", type=str, default="qwen-plus", help="Ranking model.")
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
    parser.add_argument("--rank-temperature", type=float, default=0.0, help="Ranking model temperature.")
    parser.add_argument("--rank-max-tokens", type=int, default=900, help="Ranking model max tokens.")
    parser.add_argument("--rank-concurrency", type=int, default=4, help="Concurrent ranking requests.")
    parser.add_argument("--min-candidates", type=int, default=2, help="Minimum unique candidates required for ranking.")
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count per ranking request.")
    parser.add_argument("--retry-base-sleep", type=float, default=2.0, help="Base retry sleep.")
    parser.add_argument("--retry-max-sleep", type=float, default=30.0, help="Max retry sleep.")
    parser.add_argument("--progress-every", type=int, default=25, help="Progress print interval.")
    parser.add_argument(
        "--no-reference-in-ranking",
        action="store_true",
        help="Deprecated compatibility flag. Ranking never sends the original final assistant answer to qwen-plus.",
    )
    return parser.parse_args()


def parse_float_list(raw: str, name: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def get_system_prompt(config: dict[str, Any]) -> str:
    return (config.get("prompt", {}).get("system_prompt") or "").strip()


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                records.append(json.loads(line))
    return records


def read_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in read_jsonl(path)}


def normalize_messages(example: dict[str, Any], system_prompt: str) -> tuple[list[dict[str, str]], str]:
    conversations = example.get("conversations") or []
    messages = [{"role": item["from"], "content": item["value"]} for item in conversations]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"sample {example.get('id')} does not end with assistant")

    reference = messages[-1]["content"]
    prompt_messages = messages[:-1]
    if system_prompt and (not prompt_messages or prompt_messages[0]["role"] != "system"):
        prompt_messages = [{"role": "system", "content": system_prompt}, *prompt_messages]
    return prompt_messages, reference


def load_and_sample_train(train_file: Path, sample_size: int, seed: int, system_prompt: str) -> list[dict[str, Any]]:
    rows = []
    with train_file.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            try:
                prompt_messages, reference = normalize_messages(raw, system_prompt)
            except ValueError as exc:
                print(f"Skip malformed line {line_number}: {exc}", file=sys.stderr, flush=True)
                continue
            rows.append(
                {
                    "id": str(raw.get("id") or f"line_{line_number}"),
                    "line_number": line_number,
                    "prompt_messages": prompt_messages,
                    "reference": reference,
                    "cluster_id": raw.get("cluster_id"),
                    "metadata": raw.get("metadata", {}),
                }
            )

    rng = random.Random(seed)
    rng.shuffle(rows)
    if sample_size > 0:
        rows = rows[: min(sample_size, len(rows))]
    return rows


def strip_thinking_tags(text: str) -> str:
    return re.sub(r"(?s)<think>.*?</think>", "", text).strip()


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for candidate in candidates:
        text = str(candidate.get("content", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        candidate = dict(candidate)
        candidate["index"] = len(unique)
        unique.append(candidate)
    return unique


def generate_sft_candidates(
    samples: list[dict[str, Any]],
    temperatures: list[float],
    top_ps: list[float],
    args: argparse.Namespace,
    output_path: Path,
) -> list[dict[str, Any]]:
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    existing = {} if args.overwrite else read_jsonl_by_id(output_path)
    pending = [sample for sample in samples if str(sample["id"]) not in existing]
    if args.overwrite and output_path.exists():
        output_path.unlink()
    if not pending:
        return [existing[str(sample["id"])] for sample in samples if str(sample["id"]) in existing]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    model_device = next(model.parameters()).device

    completed = dict(existing)
    batch_size = max(1, args.sft_batch_size)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        per_sample_candidates: list[list[dict[str, Any]]] = [[] for _ in batch]
        prompt_texts = [render_chat_prompt(tokenizer, sample["prompt_messages"]) for sample in batch]

        for candidate_no, (temperature, top_p) in enumerate(zip(temperatures, top_ps)):
            inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model_device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_width = inputs["input_ids"].shape[-1]
            for idx, _sample in enumerate(batch):
                output_ids = generated[idx][prompt_width:]
                content = strip_thinking_tags(tokenizer.decode(output_ids, skip_special_tokens=True))
                per_sample_candidates[idx].append(
                    {
                        "index": candidate_no,
                        "temperature": temperature,
                        "top_p": top_p,
                        "content": content,
                    }
                )

        records = []
        for idx, sample in enumerate(batch):
            candidates = dedupe_candidates(per_sample_candidates[idx])
            record = {**sample, "candidates": candidates}
            completed[str(sample["id"])] = record
            records.append(record)
        append_jsonl(output_path, records)

        done = min(start + len(batch), len(pending))
        print(f"SFT candidate progress: {done}/{len(pending)} newly generated", flush=True)

    return [completed[str(sample["id"])] for sample in samples if str(sample["id"]) in completed]


def format_messages(messages: list[dict[str, str]]) -> str:
    lines = []
    for idx, message in enumerate(messages):
        lines.append(f"{idx + 1}. {message.get('role', '')}: {message.get('content', '')}")
    return "\n".join(lines)


def format_candidates(candidates: list[dict[str, Any]]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            f"[候选 {candidate['index']} | temperature={candidate.get('temperature')} | top_p={candidate.get('top_p')}]\n"
            f"{candidate.get('content', '')}"
        )
    return "\n\n".join(blocks)


def build_ranking_prompt(sample: dict[str, Any]) -> str:
    return f"""请对下面同一电商客服对话的多个候选回复进行评分、排序，并选择best和worst。

[对话上下文]
{format_messages(sample["prompt_messages"])}

[候选回复]
{format_candidates(sample["candidates"])}

请严格按系统要求输出JSON。必须为每个候选输出scores，其中index必须来自候选index。
"""


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
        except Exception:  # noqa: BLE001
            detail = ""
        return f"HTTP Error {exc.code}: {exc.reason}; body={detail}"
    return f"{type(exc).__name__}: {exc}"


def sleep_before_retry(attempt: int, args: argparse.Namespace) -> None:
    sleep_s = min(args.retry_max_sleep, args.retry_base_sleep * (2**attempt))
    sleep_s += random.uniform(0, min(1.0, args.retry_base_sleep))
    time.sleep(max(0.0, sleep_s))


def normalize_int_range(value: Any, field_name: str, low: int, high: int) -> int:
    if isinstance(value, (int, float)) and low <= int(value) <= high:
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
        if low <= number <= high:
            return number
    raise ValueError(f"{field_name} must be {low}-{high}, got {value!r}")


def normalize_binary(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in {0, 1}:
        return int(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return int(value.strip())
    raise ValueError(f"{field_name} must be 0 or 1, got {value!r}")


def normalize_satisfaction(value: Any) -> int:
    return normalize_int_range(value, "satisfaction", 1, 5)


def normalize_overall_score(value: Any) -> float:
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        number = float(value.strip())
    else:
        raise ValueError(f"overall_score must be 0-100, got {value!r}")
    if 0.0 <= number <= 100.0:
        return round(number, 4)
    raise ValueError(f"overall_score must be 0-100, got {value!r}")


def judge_aligned_score(score: dict[str, Any]) -> float:
    if int(score["hallucination"]) == 1:
        return -100.0
    return round(
        int(score["accuracy"]) * 40.0
        + int(score["closure"]) * 25.0
        + int(score["auto_resolve"]) * 20.0
        + int(score["satisfaction"]) * 3.0
        + float(score["overall_score"]) * 0.02,
        4,
    )


def preference_key(score: dict[str, Any]) -> tuple[int, int, int, int, float, float, int]:
    return (
        int(score["accuracy"]),
        int(score["closure"]),
        int(score["auto_resolve"]),
        int(score["satisfaction"]),
        float(score["overall_score"]),
        judge_aligned_score(score),
        -int(score["index"]),
    )


def rejection_key(score: dict[str, Any]) -> tuple[int, int, int, int, float, int]:
    return (
        1 - int(score["closure"]),
        1 - int(score["auto_resolve"]),
        5 - int(score["satisfaction"]),
        100.0 - float(score["overall_score"]),
        -int(score["index"]),
    )


def normalize_ranking(raw: dict[str, Any], candidate_indexes: set[int]) -> dict[str, Any]:
    scores = []
    for item in raw.get("scores", []):
        index = int(item.get("index"))
        if index not in candidate_indexes:
            continue
        scores.append(
            {
                "index": index,
                "hallucination": normalize_binary(item.get("hallucination"), "hallucination"),
                "accuracy": normalize_binary(item.get("accuracy"), "accuracy"),
                "closure": normalize_binary(item.get("closure"), "closure"),
                "auto_resolve": normalize_binary(item.get("auto_resolve"), "auto_resolve"),
                "satisfaction": normalize_satisfaction(item.get("satisfaction")),
                "overall_score": normalize_overall_score(item.get("overall_score")),
                "rationale": str(item.get("rationale", "")).strip(),
            }
        )
    if len(scores) < 2:
        raise ValueError("ranking must contain at least two scored candidates")
    scored_indexes = {int(item["index"]) for item in scores}
    missing_indexes = candidate_indexes - scored_indexes
    if missing_indexes:
        raise ValueError(f"ranking missing scores for candidates: {sorted(missing_indexes)}")

    eligible_scores = [score for score in scores if int(score["hallucination"]) == 0]
    if not eligible_scores:
        raise ValueError("all candidates have severe hallucination=1; no valid chosen candidate")

    best = max(eligible_scores, key=preference_key)
    generic_but_correct_pool = [
        score
        for score in scores
        if int(score["index"]) != int(best["index"])
        and int(score["hallucination"]) == 0
        and int(score["accuracy"]) == 1
        and (int(score["closure"]) == 0 or int(score["auto_resolve"]) == 0)
    ]
    non_severe_rejection_pool = [
        score
        for score in scores
        if int(score["index"]) != int(best["index"]) and int(score["hallucination"]) == 0
    ]
    rejection_pool = (
        generic_but_correct_pool
        or non_severe_rejection_pool
        or [score for score in scores if int(score["index"]) != int(best["index"])]
    )
    worst = max(rejection_pool, key=rejection_key)
    best_index = int(best["index"])
    worst_index = int(worst["index"])
    if best_index == worst_index:
        raise ValueError("best_index and worst_index cannot be equal")
    best_score = judge_aligned_score(best)
    worst_score = judge_aligned_score(worst)
    score_delta = round(best_score - worst_score, 4)
    return {
        "scores": scores,
        "best_index": best_index,
        "worst_index": worst_index,
        "best_score": best_score,
        "worst_score": worst_score,
        "score_delta": score_delta,
        "selection_policy": "chosen excludes only severe hallucination=1, then sorts by accuracy, closure, auto_resolve, satisfaction, overall_score; rejected first prefers accurate non-severe but generic/unhelpful/non-closed candidates, then other non-severe candidates.",
        "rationale": str(raw.get("rationale", "")).strip(),
    }


def rank_one_sample(sample: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidates = sample.get("candidates") or []
    if len(candidates) < max(2, args.min_candidates):
        raise ValueError(f"need at least {max(2, args.min_candidates)} unique candidates")

    body = {
        "model": args.rank_model,
        "messages": [
            {"role": "system", "content": RANK_SYSTEM_PROMPT},
            {"role": "user", "content": build_ranking_prompt(sample)},
        ],
        "temperature": args.rank_temperature,
        "max_tokens": args.rank_max_tokens,
    }
    candidate_indexes = {int(candidate["index"]) for candidate in candidates}
    last_error = ""
    for attempt in range(max(1, args.max_retries)):
        try:
            payload = post_chat_completion(body, args)
            content = payload["choices"][0]["message"]["content"]
            ranking = normalize_ranking(parse_json_content(content), candidate_indexes)
            return {**sample, "ranking": ranking, "rank_model": args.rank_model}
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
                sleep_before_retry(attempt, args)
    raise RuntimeError(last_error)


def has_valid_score_ranking(record: dict[str, Any]) -> bool:
    ranking = record.get("ranking") or {}
    if "score_delta" not in ranking or "best_score" not in ranking or "worst_score" not in ranking:
        return False
    scores = ranking.get("scores") or []
    required_fields = {"hallucination", "accuracy", "closure", "auto_resolve", "satisfaction", "overall_score"}
    if not scores or not all(required_fields.issubset(score) for score in scores):
        return False
    try:
        for score in scores:
            normalize_binary(score.get("hallucination"), "hallucination")
            normalize_binary(score.get("accuracy"), "accuracy")
            normalize_binary(score.get("closure"), "closure")
            normalize_binary(score.get("auto_resolve"), "auto_resolve")
            normalize_satisfaction(score.get("satisfaction"))
            normalize_overall_score(score.get("overall_score"))
    except (TypeError, ValueError):
        return False
    return True


def score_by_index(ranking: dict[str, Any], index: int) -> dict[str, Any]:
    for score in ranking.get("scores", []):
        if int(score["index"]) == int(index):
            return score
    return {}


def build_dpo_pair_cache_record(record: dict[str, Any]) -> dict[str, Any]:
    ranking = record["ranking"]
    best_candidate = candidate_by_index(record, ranking["best_index"])
    worst_candidate = candidate_by_index(record, ranking["worst_index"])
    return {
        "id": record["id"],
        "prompt": record["prompt_messages"],
        "chosen": best_candidate["content"],
        "rejected": worst_candidate["content"],
        "ranking_brief": {
            "score_delta": ranking.get("score_delta"),
            "best_score": ranking.get("best_score"),
            "worst_score": ranking.get("worst_score"),
            "best_index": ranking.get("best_index"),
            "worst_index": ranking.get("worst_index"),
            "chosen_scores": score_by_index(ranking, ranking["best_index"]),
            "rejected_scores": score_by_index(ranking, ranking["worst_index"]),
        },
    }


def write_dpo_pair_cache(records: list[dict[str, Any]], output_path: Path) -> int:
    cache_records = []
    for record in records:
        try:
            cache_records.append(build_dpo_pair_cache_record(record))
        except (KeyError, ValueError):
            continue
    write_jsonl(output_path, cache_records)
    return len(cache_records)


def append_dpo_pair_cache(record: dict[str, Any], output_path: Path) -> bool:
    try:
        append_jsonl(output_path, [build_dpo_pair_cache_record(record)])
        return True
    except (KeyError, ValueError) as exc:
        print(f"skip rank DPO pair cache id={record.get('id')}: {exc}", file=sys.stderr, flush=True)
        return False


def rank_candidates(
    samples: list[dict[str, Any]],
    args: argparse.Namespace,
    output_path: Path,
    failed_path: Path,
    skipped_path: Path,
    pair_cache_path: Path,
) -> list[dict[str, Any]]:
    existing_all = {} if args.overwrite else read_jsonl_by_id(output_path)
    existing = {record_id: record for record_id, record in existing_all.items() if has_valid_score_ranking(record)}
    cached_ids = set() if args.overwrite else set(read_jsonl_by_id(pair_cache_path))
    min_candidates = max(2, args.min_candidates)
    pending = [
        sample
        for sample in samples
        if str(sample["id"]) not in existing and len(sample.get("candidates") or []) >= min_candidates
    ]
    skipped = [
        {
            **sample,
            "skip_reason": f"need at least {min_candidates} unique candidates",
            "num_candidates": len(sample.get("candidates") or []),
        }
        for sample in samples
        if str(sample["id"]) not in existing and len(sample.get("candidates") or []) < min_candidates
    ]
    if args.overwrite:
        for path in (output_path, failed_path, skipped_path, pair_cache_path):
            if path.exists():
                path.unlink()
    if skipped:
        write_jsonl(skipped_path, skipped)
        print(f"ranking skipped low-diversity candidates: {len(skipped)}", flush=True)
    ordered_existing = [existing[str(sample["id"])] for sample in samples if str(sample["id"]) in existing]
    missing_cache_records = [record for record in ordered_existing if str(record["id"]) not in cached_ids]
    for record in missing_cache_records:
        append_dpo_pair_cache(record, pair_cache_path)
    if not pending:
        return ordered_existing

    completed = dict(existing)
    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.rank_concurrency)) as executor:
        future_to_sample = {executor.submit(rank_one_sample, sample, args): sample for sample in pending}
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                record = future.result()
                completed[str(sample["id"])] = record
                append_jsonl(output_path, [record])
                append_dpo_pair_cache(record, pair_cache_path)
            except Exception as exc:  # noqa: BLE001
                failed_record = {**sample, "error": format_exception(exc)}
                failed.append(failed_record)
                append_jsonl(failed_path, [failed_record])
                print(f"ranking failed id={sample['id']}: {failed_record['error']}", file=sys.stderr, flush=True)

            done = len(completed) - len(existing) + len(failed)
            progress_every = max(1, args.progress_every)
            if done % progress_every == 0 or done == len(pending):
                print(
                    f"ranking progress: {done}/{len(pending)} attempted, "
                    f"succeeded={len(completed) - len(existing)}, failed={len(failed)}",
                    flush=True,
                )

    return [completed[str(sample["id"])] for sample in samples if str(sample["id"]) in completed]


def candidate_by_index(record: dict[str, Any], index: int) -> dict[str, Any]:
    for candidate in record.get("candidates", []):
        if int(candidate["index"]) == int(index):
            return candidate
    raise KeyError(f"candidate index not found: {index}")


def build_dpo_record(record: dict[str, Any]) -> dict[str, Any]:
    ranking = record["ranking"]
    best_candidate = candidate_by_index(record, ranking["best_index"])
    worst_candidate = candidate_by_index(record, ranking["worst_index"])
    return {
        "id": record["id"],
        "prompt": record["prompt_messages"],
        "chosen": [{"role": "assistant", "content": best_candidate["content"]}],
        "rejected": [{"role": "assistant", "content": worst_candidate["content"]}],
        "reference": record.get("reference", ""),
        "metadata": {
            "line_number": record.get("line_number"),
            "cluster_id": record.get("cluster_id"),
            "source_metadata": record.get("metadata", {}),
            "rank_model": record.get("rank_model"),
            "ranking": ranking,
            "score_delta": ranking.get("score_delta"),
            "chosen_candidate": {
                "index": best_candidate["index"],
                "temperature": best_candidate.get("temperature"),
                "top_p": best_candidate.get("top_p"),
            },
            "rejected_candidate": {
                "index": worst_candidate["index"],
                "temperature": worst_candidate.get("temperature"),
                "top_p": worst_candidate.get("top_p"),
            },
        },
    }


def compute_delta_stats(deltas: list[float]) -> dict[str, Any]:
    if not deltas:
        return {}
    ordered = sorted(deltas)
    count = len(ordered)

    def percentile(pct: float) -> float:
        idx = min(count - 1, max(0, int((count - 1) * pct)))
        return round(ordered[idx], 4)

    return {
        "min": round(ordered[0], 4),
        "p10": percentile(0.10),
        "p30": percentile(0.30),
        "median": percentile(0.50),
        "p70": percentile(0.70),
        "p90": percentile(0.90),
        "max": round(ordered[-1], 4),
        "mean": round(sum(ordered) / count, 4),
    }


def save_dpo_pairs(
    records: list[dict[str, Any]],
    output_path: Path,
    dropped_output_path: Path,
    summary_path: Path,
    discard_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_pairs = []
    skipped = 0
    for record in records:
        try:
            pair = build_dpo_record(record)
        except (KeyError, ValueError) as exc:
            print(f"skip DPO pair id={record.get('id')}: {exc}", file=sys.stderr, flush=True)
            skipped += 1
            continue
        if pair["chosen"][0]["content"].strip() == pair["rejected"][0]["content"].strip():
            skipped += 1
            continue
        if pair.get("metadata", {}).get("score_delta") is None:
            skipped += 1
            continue
        raw_pairs.append(pair)

    discard_fraction = min(1.0, max(0.0, discard_fraction))
    drop_count = int(round(len(raw_pairs) * discard_fraction))
    sorted_pairs = sorted(
        raw_pairs,
        key=lambda item: (
            float(item.get("metadata", {}).get("score_delta", 0.0)),
            item.get("id", ""),
        ),
    )
    dropped_pairs = sorted_pairs[:drop_count]
    kept_pairs = sorted_pairs[drop_count:]
    threshold = None
    if dropped_pairs:
        threshold = round(float(dropped_pairs[-1].get("metadata", {}).get("score_delta", 0.0)), 4)

    for pair in kept_pairs:
        pair["metadata"]["quality_filter"] = {
            "kept": True,
            "discard_fraction": discard_fraction,
            "drop_threshold": threshold,
            "reason": "kept_by_score_delta",
        }
    for pair in dropped_pairs:
        pair["metadata"]["quality_filter"] = {
            "kept": False,
            "discard_fraction": discard_fraction,
            "drop_threshold": threshold,
            "reason": "low_score_delta",
        }

    kept_pairs = sorted(kept_pairs, key=lambda item: item.get("id", ""))
    dropped_pairs = sorted(dropped_pairs, key=lambda item: item.get("id", ""))
    deltas = [float(pair.get("metadata", {}).get("score_delta", 0.0)) for pair in raw_pairs]
    summary = {
        "ranked_records": len(records),
        "raw_pairs": len(raw_pairs),
        "kept_pairs": len(kept_pairs),
        "dropped_pairs": len(dropped_pairs),
        "skipped_pairs": skipped,
        "discard_fraction": discard_fraction,
        "drop_threshold": threshold,
        "delta_stats": compute_delta_stats(deltas),
        "note": "Pairs with the smallest best_score - worst_score are dropped to remove weak preference signals.",
    }
    write_jsonl(output_path, kept_pairs)
    write_jsonl(dropped_output_path, dropped_pairs)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    return kept_pairs, summary


def main() -> None:
    args = parse_args()
    temperatures = parse_float_list(args.temperatures, "temperatures")
    top_ps = parse_float_list(args.top_ps, "top_ps")
    if len(top_ps) == 1:
        top_ps = top_ps * len(temperatures)
    if len(top_ps) != len(temperatures):
        raise ValueError("--top-ps must have length 1 or match --temperatures")
    if len(temperatures) < 2:
        raise ValueError("at least two temperatures are required to build preference pairs")

    output_dir = Path(args.output_dir).resolve()
    sampled_path = output_dir / "sampled_prompts.jsonl"
    candidates_path = output_dir / "sft_candidates.jsonl"
    ranked_path = output_dir / "qwen_ranked.jsonl"
    failed_rankings_path = output_dir / "failed_rankings.jsonl"
    skipped_rankings_path = output_dir / "skipped_low_diversity_rankings.jsonl"
    rank_pair_cache_path = output_dir / "rank_dpo_pair_cache.jsonl"
    dpo_path = output_dir / "dpo_pairs.jsonl"
    dropped_dpo_path = output_dir / "dropped_dpo_pairs.jsonl"
    quality_summary_path = output_dir / "pair_quality_summary.json"

    config = load_config(Path(args.config).resolve())
    if not args.overwrite and sampled_path.exists():
        samples = read_jsonl(sampled_path)
    else:
        samples = load_and_sample_train(
            Path(args.train_file).resolve(),
            args.sample_size,
            args.seed,
            get_system_prompt(config),
        )
        write_jsonl(sampled_path, samples)

    print(
        json.dumps(
            {
                "sampled": len(samples),
                "temperatures": temperatures,
                "top_ps": top_ps,
                "sampled_output": str(sampled_path),
                "sft_candidates_output": str(candidates_path),
                "qwen_ranked_output": str(ranked_path),
                "skipped_rankings_output": str(skipped_rankings_path),
                "rank_dpo_pair_cache_output": str(rank_pair_cache_path),
                "dpo_output": str(dpo_path),
                "dropped_dpo_output": str(dropped_dpo_path),
                "pair_quality_summary": str(quality_summary_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if args.skip_sft:
        by_id = read_jsonl_by_id(candidates_path)
        candidate_records = [by_id[str(sample["id"])] for sample in samples if str(sample["id"]) in by_id]
    else:
        candidate_records = generate_sft_candidates(samples, temperatures, top_ps, args, candidates_path)

    if args.skip_ranking:
        by_id = read_jsonl_by_id(ranked_path)
        ranked_records = [
            by_id[str(record["id"])]
            for record in candidate_records
            if str(record["id"]) in by_id and has_valid_score_ranking(by_id[str(record["id"])])
        ]
        cache_count = write_dpo_pair_cache(ranked_records, rank_pair_cache_path)
        print(f"rank DPO pair cache refreshed: {cache_count} -> {rank_pair_cache_path}", flush=True)
    else:
        ranked_records = rank_candidates(
            candidate_records,
            args,
            ranked_path,
            failed_rankings_path,
            skipped_rankings_path,
            rank_pair_cache_path,
        )

    pairs, quality_summary = save_dpo_pairs(
        ranked_records,
        dpo_path,
        dropped_dpo_path,
        quality_summary_path,
        args.discard_fraction,
    )
    print(
        json.dumps(
            {
                "sampled": len(samples),
                "candidate_records": len(candidate_records),
                "ranked_records": len(ranked_records),
                "dpo_pairs": len(pairs),
                "rank_dpo_pair_cache_output": str(rank_pair_cache_path),
                "dpo_output": str(dpo_path),
                "dropped_dpo_output": str(dropped_dpo_path),
                "pair_quality_summary": quality_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
