#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""清洗网店客服问答 Excel，并导出适合 SFT 的 ShareGPT 风格 jsonl。

默认流程：
1. 读取 `dataprocess/网店客服回复数据集.xlsx`
2. 对 `问题` / `回复` 做基础文本清洗
3. 过滤空文本、极短文本、低价值模板文本
4. 先做 exact dedup，再做 fuzzy dedup
5. 导出 `all/train/val` jsonl 与审计报告
"""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = SCRIPT_DIR / "网店客服回复数据集.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT  / "data"
DEFAULT_AUDIT_DIR = SCRIPT_DIR / "audit"

TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\u0000-\u0008\u000B-\u001F\u007F-\u009F]")
ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\uFEFF]")
REPLACEMENT_CHAR_RE = re.compile(r"�+")
REPEATED_PUNCT_RE = re.compile(r"([,，.。!！?？~～:：;；、])\1+")
LEADING_TRAILING_PUNCT_RE = re.compile(r"^[\s,，.。!！?？~～:：;；、/\\|]+|[\s,，.。!！?？~～:：;；、/\\|]+$")
MEANINGLESS_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
DATE_YMD_CN_RE = re.compile(r"(?<!\d)(20\d{2})\s*[-/\.年]\s*(\d{1,2})\s*[-/\.月]\s*(\d{1,2})\s*日?(?!\d)")
DATE_MD_CN_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[-/\.月]\s*(\d{1,2})\s*日(?!\d)")
TIME_COLON_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[:：]\s*(\d{1,2})(?!\d)")
TIME_CN_RE = re.compile(r"(?<!\d)(\d{1,2})\s*点\s*(\d{1,2})?\s*分?(?!\d)")
NUMBER_WITH_COMMA_RE = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+)(?:\.(\d+))?(?!\d)")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)

PUNCT_TRANSLATION = str.maketrans(
    {
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
        "《": "<",
        "》": ">",
        "—": "-",
        "－": "-",
        "–": "-",
        "·": " ",
        "•": " ",
        "…": "...",
        "￥": "¥",
        "／": "/",
        "＼": "\\",
        "　": " ",
    }
)

LOW_VALUE_QUESTIONS = {
    "在吗",
    "在不在",
    "有人吗",
    "你好",
    "您好",
    "好的",
    "好",
    "嗯",
    "哦",
    "收到",
}

LOW_VALUE_ANSWERS = {
    "好的",
    "好",
    "嗯",
    "哦",
    "亲",
    "亲亲",
    "您好",
    "你好",
    "在的",
    "在呢",
    "稍等",
    "稍等哈",
    "请稍等",
    "不客气",
    "谢谢",
}


@dataclass
class Sample:
    idx: int
    question_raw: str
    answer_raw: str
    question_clean: str
    answer_clean: str
    question_norm: str
    answer_norm: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean customer-service QA dataset for SFT.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Path to xlsx input file.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for jsonl outputs.")
    parser.add_argument("--audit-dir", type=str, default=str(DEFAULT_AUDIT_DIR), help="Directory for audit files.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--min-question-chars", type=int, default=2, help="Minimum cleaned question length.")
    parser.add_argument("--min-answer-chars", type=int, default=4, help="Minimum cleaned answer length.")
    parser.add_argument("--fuzzy-question-threshold", type=float, default=0.96, help="Question similarity threshold.")
    parser.add_argument("--fuzzy-answer-threshold", type=float, default=0.92, help="Answer similarity threshold.")
    parser.add_argument("--fuzzy-pair-threshold", type=float, default=0.95, help="Full pair similarity threshold.")
    parser.add_argument("--audit-limit", type=int, default=3000, help="Max rows per audit file.")
    return parser.parse_args()


def safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def standardize_date(match: re.Match[str]) -> str:
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    return f"{year:04d}-{month:02d}-{day:02d}"


def standardize_md_date(match: re.Match[str]) -> str:
    month = int(match.group(1))
    day = int(match.group(2))
    return f"{month:02d}-{day:02d}"


def standardize_colon_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    return f"{hour:02d}:{minute:02d}"


def standardize_cn_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    return f"{hour:02d}:{minute:02d}"


def standardize_number(match: re.Match[str]) -> str:
    integer_part = match.group(1).replace(",", "")
    decimal_part = match.group(2)
    if decimal_part is not None:
        return f"{integer_part}.{decimal_part}"
    return integer_part


def strip_emoji(text: str) -> str:
    return EMOJI_RE.sub(" ", text)


def clean_text(text: str) -> str:
    text = safe_text(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(PUNCT_TRANSLATION)
    text = TAG_RE.sub(" ", text)
    text = strip_emoji(text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = CONTROL_RE.sub(" ", text)
    text = REPLACEMENT_CHAR_RE.sub(" ", text)
    text = DATE_YMD_CN_RE.sub(standardize_date, text)
    text = DATE_MD_CN_RE.sub(standardize_md_date, text)
    text = TIME_COLON_RE.sub(standardize_colon_time, text)
    text = TIME_CN_RE.sub(standardize_cn_time, text)
    text = NUMBER_WITH_COMMA_RE.sub(standardize_number, text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE_RE.sub(" ", text.replace("\n", " ")).strip()
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = re.sub(r"\s*([,，.。!！?？:：;；、])\s*", r"\1", text)
    text = re.sub(r"([。！？])([^\s])", r"\1 \2", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_for_dedup(text: str) -> str:
    text = clean_text(text).lower()
    text = LEADING_TRAILING_PUNCT_RE.sub("", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，,。.!！?？:：;；、/\\|()\[\]{}\"'`~\-]+", "", text)
    return text


def is_low_value_question(text: str, min_chars: int) -> bool:
    core = LEADING_TRAILING_PUNCT_RE.sub("", text)
    if not core:
        return True
    if len(core) < min_chars:
        return True
    if core in LOW_VALUE_QUESTIONS:
        return True
    if MEANINGLESS_ONLY_RE.fullmatch(core):
        return True
    return False


def is_low_value_answer(text: str, min_chars: int) -> bool:
    core = LEADING_TRAILING_PUNCT_RE.sub("", text)
    if not core:
        return True
    if len(core) < min_chars:
        return True
    if core in LOW_VALUE_ANSWERS:
        return True
    if MEANINGLESS_ONLY_RE.fullmatch(core):
        return True
    return False


def quality_score(sample: Sample) -> tuple[int, int, int]:
    return (
        len(sample.answer_clean),
        len(sample.question_clean),
        -sample.idx,
    )


def append_limited(rows: list[dict], item: dict, limit: int) -> None:
    if len(rows) < limit:
        rows.append(item)


def pair_to_record(sample: Sample) -> dict:
    return {
        "id": f"customer_service_{sample.idx:06d}",
        "conversations": [
            {"from": "user", "value": sample.question_clean},
            {"from": "assistant", "value": sample.answer_clean},
        ],
    }


def export_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_samples(args: argparse.Namespace) -> tuple[list[Sample], dict[str, list[dict]]]:
    df = pd.read_excel(args.input)
    if "问题" not in df.columns or "回复" not in df.columns:
        raise ValueError("Input xlsx must contain columns: 问题, 回复")

    audit = {
        "dropped_empty_or_short": [],
        "dropped_low_value": [],
    }
    stats = {
        "rows_raw": len(df),
        "dropped_empty_or_short": 0,
        "dropped_low_value": 0,
    }
    samples: list[Sample] = []

    for idx, row in df.iterrows():
        question_raw = safe_text(row["问题"])
        answer_raw = safe_text(row["回复"])
        question_clean = clean_text(question_raw)
        answer_clean = clean_text(answer_raw)

        if not question_clean or not answer_clean:
            stats["dropped_empty_or_short"] += 1
            append_limited(
                audit["dropped_empty_or_short"],
                {
                    "row_idx": int(idx),
                    "question_raw": question_raw,
                    "answer_raw": answer_raw,
                    "question_clean": question_clean,
                    "answer_clean": answer_clean,
                    "reason": "empty_after_clean",
                },
                args.audit_limit,
            )
            continue

        if is_low_value_question(question_clean, args.min_question_chars) or is_low_value_answer(
            answer_clean, args.min_answer_chars
        ):
            stats["dropped_low_value"] += 1
            append_limited(
                audit["dropped_low_value"],
                {
                    "row_idx": int(idx),
                    "question_raw": question_raw,
                    "answer_raw": answer_raw,
                    "question_clean": question_clean,
                    "answer_clean": answer_clean,
                    "reason": "low_value_or_too_short",
                },
                args.audit_limit,
            )
            continue

        samples.append(
            Sample(
                idx=int(idx),
                question_raw=question_raw,
                answer_raw=answer_raw,
                question_clean=question_clean,
                answer_clean=answer_clean,
                question_norm=normalize_for_dedup(question_clean),
                answer_norm=normalize_for_dedup(answer_clean),
            )
        )

    audit["stats"] = stats
    return samples, audit


def exact_dedup(samples: list[Sample], audit_limit: int) -> tuple[list[Sample], list[dict], int]:
    kept: list[Sample] = []
    dropped: list[dict] = []
    dropped_count = 0
    best_by_key: dict[tuple[str, str], Sample] = {}

    for sample in sorted(samples, key=quality_score, reverse=True):
        key = (sample.question_norm, sample.answer_norm)
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = sample
            kept.append(sample)
            continue

        dropped_count += 1
        append_limited(
            dropped,
            {
                "row_idx": sample.idx,
                "duplicate_of": existing.idx,
                "question_clean": sample.question_clean,
                "answer_clean": sample.answer_clean,
                "reason": "exact_duplicate",
            },
            audit_limit,
        )

    kept.sort(key=lambda item: item.idx)
    return kept, dropped, dropped_count


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def bucket_keys(text: str) -> set[str]:
    core = text or ""
    if not core:
        return {"empty"}
    length_band = str(min(len(core) // 4, 50))
    prefix = core[:8]
    suffix = core[-8:]
    return {
        f"p:{length_band}:{prefix}",
        f"s:{length_band}:{suffix}",
    }


def fuzzy_dedup(samples: list[Sample], args: argparse.Namespace) -> tuple[list[Sample], list[dict], int]:
    buckets: dict[str, list[Sample]] = defaultdict(list)
    kept: list[Sample] = []
    dropped: list[dict] = []
    dropped_count = 0

    for sample in sorted(samples, key=quality_score, reverse=True):
        candidates: list[Sample] = []
        seen_candidate_ids: set[int] = set()
        for key in bucket_keys(sample.question_norm):
            for candidate in buckets.get(key, []):
                if candidate.idx not in seen_candidate_ids:
                    candidates.append(candidate)
                    seen_candidate_ids.add(candidate.idx)

        duplicate_of: Sample | None = None
        best_scores: dict[str, float] | None = None

        for candidate in candidates:
            q_sim = similarity(sample.question_norm, candidate.question_norm)
            if q_sim < args.fuzzy_question_threshold:
                continue

            a_sim = similarity(sample.answer_norm, candidate.answer_norm)
            pair_sim = similarity(
                f"{sample.question_norm}||{sample.answer_norm}",
                f"{candidate.question_norm}||{candidate.answer_norm}",
            )
            if a_sim >= args.fuzzy_answer_threshold or pair_sim >= args.fuzzy_pair_threshold:
                duplicate_of = candidate
                best_scores = {
                    "question_similarity": round(q_sim, 4),
                    "answer_similarity": round(a_sim, 4),
                    "pair_similarity": round(pair_sim, 4),
                }
                break

        if duplicate_of is not None:
            dropped_count += 1
            append_limited(
                dropped,
                {
                    "row_idx": sample.idx,
                    "duplicate_of": duplicate_of.idx,
                    "question_clean": sample.question_clean,
                    "answer_clean": sample.answer_clean,
                    "reason": "fuzzy_duplicate",
                    **(best_scores or {}),
                },
                args.audit_limit,
            )
            continue

        kept.append(sample)
        for key in bucket_keys(sample.question_norm):
            buckets[key].append(sample)

    kept.sort(key=lambda item: item.idx)
    return kept, dropped, dropped_count


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio)) if shuffled else 0
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    return train_records, val_records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    samples, base_audit = load_samples(args)
    after_exact, exact_dropped, exact_dropped_count = exact_dedup(samples, args.audit_limit)
    after_fuzzy, fuzzy_dropped, fuzzy_dropped_count = fuzzy_dedup(after_exact, args)
    records = [pair_to_record(sample) for sample in after_fuzzy]
    train_records, val_records = split_records(records, args.val_ratio, args.seed)

    export_jsonl(output_dir / "all.jsonl", records)
    export_jsonl(output_dir / "train.jsonl", train_records)
    export_jsonl(output_dir / "val.jsonl", val_records)
    export_jsonl(audit_dir / "audit_dropped_empty_or_short.jsonl", base_audit["dropped_empty_or_short"])
    export_jsonl(audit_dir / "audit_dropped_low_value.jsonl", base_audit["dropped_low_value"])
    export_jsonl(audit_dir / "audit_removed_exact_duplicates.jsonl", exact_dropped)
    export_jsonl(audit_dir / "audit_removed_fuzzy_duplicates.jsonl", fuzzy_dropped)

    report = {
        "input_path": str(Path(args.input).resolve()),
        "output_dir": str(output_dir.resolve()),
        "audit_dir": str(audit_dir.resolve()),
        "rows_raw": base_audit["stats"]["rows_raw"],
        "rows_after_clean_filter": len(samples),
        "rows_after_exact_dedup": len(after_exact),
        "rows_after_fuzzy_dedup": len(after_fuzzy),
        "train_size": len(train_records),
        "val_size": len(val_records),
        "dropped_empty_or_short": base_audit["stats"]["dropped_empty_or_short"],
        "dropped_low_value": base_audit["stats"]["dropped_low_value"],
        "removed_exact_duplicates": exact_dropped_count,
        "removed_fuzzy_duplicates": fuzzy_dropped_count,
        "audit_exported_rows": {
            "dropped_empty_or_short": len(base_audit["dropped_empty_or_short"]),
            "dropped_low_value": len(base_audit["dropped_low_value"]),
            "removed_exact_duplicates": len(exact_dropped),
            "removed_fuzzy_duplicates": len(fuzzy_dropped),
        },
        "top_question_lengths": Counter(min(len(item.question_clean) // 10 * 10, 100) for item in after_fuzzy).most_common(10),
        "top_answer_lengths": Counter(min(len(item.answer_clean) // 20 * 20, 200) for item in after_fuzzy).most_common(10),
        "config": {
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "min_question_chars": args.min_question_chars,
            "min_answer_chars": args.min_answer_chars,
            "fuzzy_question_threshold": args.fuzzy_question_threshold,
            "fuzzy_answer_threshold": args.fuzzy_answer_threshold,
            "fuzzy_pair_threshold": args.fuzzy_pair_threshold,
        },
    }
    (audit_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    preview_rows = [
        {
            "question": sample.question_clean,
            "answer": sample.answer_clean,
        }
        for sample in after_fuzzy[:20]
    ]
    (audit_dir / "preview_clean_samples.json").write_text(
        json.dumps(preview_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
