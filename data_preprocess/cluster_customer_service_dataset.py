#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""将基础清洗后的客服 SFT 数据做问题聚类、回答重排与矛盾剔除。

Pipeline
1. 读取 ShareGPT 风格 `jsonl`
2. 对问题做 embedding / tfidf 表征
3. 用 FAISS 或 sklearn 近邻搜索构图并聚类
4. 每个问题簇内对回答做家族聚合、质量排序、矛盾答案剔除
5. 导出更稳定的高质量 SFT 分布
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize

from SFT_DPO.dataprocess.clean_customer_service_dataset import clean_text, normalize_for_dedup


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT = REPO_ROOT / "Main" / "data" / "all.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Main" / "data_clustered"
DEFAULT_AUDIT_DIR = SCRIPT_DIR / "cluster_audit"

YESNO_HINTS = ("是否", "能否", "可否", "可以", "能不能", "支不支持", "有没有", "是不是", "包邮", "正品", "发票", "货到付款")
TIME_HINTS = ("多久", "几天", "何时", "什么时候", "时间", "速度", "发货时间", "到货", "送达", "时效", "多长时间")
MONEY_HINTS = ("价格", "便宜", "优惠", "折扣", "满减", "包邮", "运费")
RETURN_HINTS = ("退货", "换货", "退换货", "售后", "退款", "保修")
QUESTION_TOPIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "invoice": ("发票", "开票"),
    "cod": ("货到付款",),
    "custom": ("定制", "定制化", "个性化"),
    "contact": ("联系客服", "联系你们", "联系你", "联系商家", "怎么联系", "如何联系"),
    "service_hours": ("工作时间", "服务时间", "在线时间", "营业时间", "客服时间"),
    "logistics_query": ("物流信息", "查询物流", "物流查询", "物流进度", "物流状态", "物流更新", "查看物流", "快递单号", "物流单号", "跟踪包裹", "追踪包裹"),
    "damaged_goods": ("商品损坏", "商品破损", "货品损坏", "货品破损", "收到商品有损坏", "收到商品损坏", "收到商品破损", "收到货品有损坏", "收到货品损坏", "收到货品破损", "收到货有损坏", "收到货损坏", "收到货破损", "包裹破损", "包裹损坏"),
    "logistics_issue": ("物流问题", "物流出现问题", "物流有问题", "物流异常", "物流延误", "包裹延误", "包裹丢失", "包裹损坏", "运输问题", "配送问题", "快递问题"),
    "pickup": ("自提", "自取", "自提点", "到店取", "到店自提", "门店自提", "仓库自提", "自行取", "自行领取", "上门自取", "取货点"),
    "delivery_schedule": ("指定配送时间", "指定送货时间", "预约配送时间", "预约送货时间", "选择配送时间", "选择送货时间", "更改配送时间", "修改配送时间", "定制配送时间", "特定配送时间", "特定的配送时间", "特定送货时间", "特定的送货时间", "固定配送时间", "固定送货时间", "指定时间", "配送时间限制", "物流配送有时间限制"),
    "international_delivery_time": ("国际物流", "国际快递", "海外配送", "跨境配送", "海外物流", "跨境物流"),
    "ship_time": ("发货时间", "发货多久", "多久发货", "多久可以发货", "多久能发货", "何时发货", "何时可以发货", "何时能发货", "什么时候发货", "什么时候能发货", "什么时候可以发货", "什么时候发出", "什么时候寄出", "发货速度", "发货快", "发货快吗", "几天发货", "几天可以发货", "发货需要多长时间", "发货多长时间", "出库时间", "订单处理时间"),
    "delivery_time": ("到货", "送达", "送到", "多久到", "多久可以到", "多久能到", "多久送到", "多久可以送到", "多久能送到", "多久收到", "多久能收到", "多久可以收到", "多久可以收到商品", "多久可以收到货", "几天能到", "几天收到", "几天可以到", "几天可以送到", "几天能送到", "什么时候能到", "什么时候到", "什么时候收到", "何时到", "何时能到", "何时收到", "何时可以收到", "多长时间到", "多长时间收到", "配送时间", "送货时间", "交货时间", "运输时间", "物流配送时间", "物流配送的时间", "物流配送具体时间", "物流到达时间", "预计物流到达时间", "配送时间估算", "物流配送时间估算", "配送速度", "送货速度", "配送快", "配送快吗", "送货快", "物流多久", "物流时间", "物流速度", "物流运输时间", "物流运输需要多久", "物流是否快速", "物流快", "物流快吗", "快递速度", "快递有多快", "快递多快", "快递快", "快递快吗", "快递几天", "快递通常需要多久", "快递通常需要多长时间", "几天到"),
    "return": ("退货", "换货", "退换货", "退款", "退货流程", "退换货流程", "退换货政策"),
    "aftersales": ("售后",),
    "warranty": ("保修", "质保"),
    "stock": ("有货", "现货", "库存", "缺货"),
    "payment": ("支付方式", "付款方式", "怎么支付", "如何支付", "支付渠道"),
    "address_change": ("修改收货地址", "更改收货地址", "修改配送地址", "更改配送地址"),
}

POSITIVE_PATTERNS = (
    "支持",
    "可以",
    "能够",
    "能",
    "提供",
    "接受",
    "允许",
    "包邮",
    "正品",
    "现货",
    "在售",
)
NEGATIVE_PATTERNS = (
    "不支持",
    "暂不支持",
    "不能",
    "不可以",
    "没有",
    "不是",
    "不提供",
    "不接受",
    "不允许",
    "不包邮",
    "无货",
    "缺货",
    "售罄",
    "暂时不",
)

FILLER_TERMS = (
    "亲亲",
    "亲",
    "呢",
    "哦",
    "呀",
    "哈",
    "小主",
    "非常抱歉",
    "感谢您的理解",
    "感谢您的支持",
    "您可以放心",
)
QUALITY_POSITIVE_TERMS = ("请", "建议", "可以", "支持", "提供", "具体", "联系客服", "订单", "物流", "发票", "退换货")
RISKY_PROMISE_TERMS = ("终身保修", "72小时内送达", "100%", "绝对", "全国都", "一定到", "永久")
CONDITIONAL_TERMS = ("如果", "若", "一旦", "具体", "视情况", "部分", "根据", "未发货前", "发货后", "以实际", "取决于")
WORKDAY_RE = re.compile(r"(?<!\d)(\d{1,2})(?:\s*[-~到至]\s*(\d{1,2}))?\s*(个?工作日|天|日|小时|h)\s*(以内|之内|内|左右|上下)?(?!\d)")
DIGIT_RE = re.compile(r"\d+")
TIME_POLICY_TYPES = {"ship_time", "delivery_time", "international_delivery_time"}
TIME_POLICY_WINDOWS = {
    "ship_time": ("1个工作日", "3个工作日"),
    "delivery_time": ("3个工作日", "7个工作日"),
}
CONSERVATIVE_POLICY_TYPES = {"pickup", "damaged_goods"}
LARGE_CLUSTER_FAST_FAMILY_THRESHOLD = 100


@dataclass
class QARecord:
    idx: int
    source_id: str
    question: str
    answer: str
    question_norm: str
    answer_norm: str


@dataclass
class AnswerFamily:
    family_id: int
    representative: QARecord
    members: list[QARecord]
    support_count: int
    quality_score: float
    stance: str
    time_ranges: list[tuple[float, float]]
    contradiction_reason: str | None = None


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster similar customer-service questions and stabilize answer distribution.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Input ShareGPT jsonl.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for filtered jsonl.")
    parser.add_argument("--audit-dir", type=str, default=str(DEFAULT_AUDIT_DIR), help="Audit directory.")
    parser.add_argument("--embedding-backend", type=str, default="sentence-transformers", choices=["auto", "sentence-transformers", "transformers", "tfidf"], help="Question embedding backend.")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-small-zh-v1.5", help="Embedding model path or name.")
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"], help="Device for embedding inference.")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size.")
    parser.add_argument("--ann-backend", type=str, default="faiss", choices=["auto", "faiss", "sklearn"], help="Approximate nearest-neighbor backend.")
    parser.add_argument("--neighbors-k", type=int, default=25, help="Nearest neighbors to inspect per question.")
    parser.add_argument("--similarity-threshold", type=float, default=-1.0, help="Question similarity threshold. Negative means auto.")
    parser.add_argument("--answer-family-threshold", type=float, default=0.82, help="Within-cluster answer family merge threshold.")
    parser.add_argument("--max-answers-per-cluster", type=int, default=2, help="Max kept answers per cluster after filtering.")
    parser.add_argument("--unify-time-answers", action=argparse.BooleanOptionalAction, default=True, help="Rewrite time-sensitive clusters into one conservative policy answer.")
    parser.add_argument("--time-policy-upper-percentile", type=float, default=0.85, help="Upper percentile used when consolidating noisy time ranges.")
    parser.add_argument("--max-question-variants-per-answer", type=int, default=4, help="Max original question variants emitted per kept answer. 1 means canonical only.")
    parser.add_argument("--min-family-support", type=int, default=1, help="Minimum members for an answer family to be kept.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for split.")
    parser.add_argument("--augment-ratio", type=float, default=0.0, help="Per-cluster ratio of clustered FAQ rows to rewrite into multi-turn dialogues. 0 disables OpenAI calls.")
    parser.add_argument(
        "--augment-sampling-strategy",
        type=str,
        default="coverage",
        choices=["coverage", "per-cluster-ratio"],
        help="Augmentation sampling strategy. coverage keeps the total ratio and touches as many clusters as possible.",
    )
    parser.add_argument("--augment-min-per-cluster", type=int, default=0, help="Minimum sampled rows per cluster when augmentation is enabled. Use 1 to force every non-empty cluster to have at least one rewritten row.")
    parser.add_argument("--augment-model", type=str, default=os.environ.get("OPENAI_MODEL", "qwen-plus-latest"), help="OpenAI model used for FAQ-to-dialogue rewriting.")
    parser.add_argument("--augment-api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable that stores the OpenAI API key.")
    parser.add_argument("--augment-api-base", type=str, default=os.environ.get("OPENAI_BASE_URL", ""), help="OpenAI-compatible API base URL. Defaults to the OPENAI_BASE_URL environment variable.")
    parser.add_argument("--augment-cache", type=str, default="", help="JSONL cache path for generated dialogues. Defaults to audit_dir/faq_dialogue_augmentation_cache.jsonl.")
    parser.add_argument("--augment-max-retries", type=int, default=3, help="Retries per sampled row when JSON validation fails or the API request errors.")
    parser.add_argument("--augment-concurrency", type=int, default=8, help="Concurrent OpenAI requests used during augmentation.")
    parser.add_argument("--augment-timeout", type=float, default=60.0, help="OpenAI request timeout in seconds.")
    return parser.parse_args()


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def sequence_ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def load_records(path: str) -> list[QARecord]:
    records: list[QARecord] = []
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for idx, line in enumerate(file_obj):
            obj = json.loads(line)
            conv = obj["conversations"]
            question = clean_text(conv[0]["value"])
            answer = clean_text(conv[1]["value"])
            records.append(
                QARecord(
                    idx=idx,
                    source_id=obj.get("id", f"sample_{idx:06d}"),
                    question=question,
                    answer=answer,
                    question_norm=normalize_for_dedup(question),
                    answer_norm=normalize_for_dedup(answer),
                )
            )
    return records


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return "cuda"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def embed_with_sentence_transformers(texts: list[str], model_name: str, device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def embed_with_transformers(texts: list[str], model_name: str, device: str, batch_size: int) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model_out = model(**encoded)
            if hasattr(model_out, "last_hidden_state"):
                pooled = mean_pool(model_out.last_hidden_state, encoded["attention_mask"])
            elif isinstance(model_out, tuple):
                pooled = mean_pool(model_out[0], encoded["attention_mask"])
            else:
                pooled = model_out.pooler_output
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(outputs)


def embed_with_tfidf(texts: list[str]) -> tuple[Any, str]:
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
    matrix = vectorizer.fit_transform(texts)
    matrix = sk_normalize(matrix)
    return matrix, "tfidf-char-2-4"


def build_embeddings(records: list[QARecord], args: argparse.Namespace) -> tuple[Any, str]:
    texts = [record.question for record in records]
    backend = args.embedding_backend
    device = resolve_device(args.device)
    errors: list[str] = []

    if backend in ("auto", "sentence-transformers"):
        try:
            embeddings = embed_with_sentence_transformers(texts, args.embedding_model, device=device, batch_size=args.batch_size)
            return embeddings, f"sentence-transformers:{device}"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"sentence-transformers failed: {exc}")
            if backend == "sentence-transformers":
                raise

    if backend in ("auto", "transformers"):
        try:
            embeddings = embed_with_transformers(texts, args.embedding_model, device=device, batch_size=max(8, min(args.batch_size, 64)))
            return embeddings, f"transformers:{device}"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"transformers failed: {exc}")
            if backend == "transformers":
                raise

    matrix, name = embed_with_tfidf(texts)
    if errors:
        print("Embedding fallback:", " | ".join(errors))
    return matrix, name


def default_similarity_threshold(embedding_name: str) -> float:
    if embedding_name.startswith("tfidf"):
        return 0.72
    return 0.83


def question_topics(question: str) -> set[str]:
    topics: set[str] = set()
    for topic, patterns in QUESTION_TOPIC_PATTERNS.items():
        if any(pattern in question for pattern in patterns):
            topics.add(topic)
    if any(token in question for token in TIME_HINTS):
        if any(token in question for token in ("发货", "发出", "寄出", "出库")):
            topics.add("ship_time")
        if any(token in question for token in ("到货", "送达", "送到", "收到", "收货", "交货", "运输", "快递")):
            topics.add("delivery_time")
    if not topics:
        if any(token in question for token in RETURN_HINTS):
            topics.add("return")
        elif any(token in question for token in TIME_HINTS):
            topics.add("time_generic")
        elif any(token in question for token in MONEY_HINTS):
            topics.add("money")
        elif any(token in question for token in YESNO_HINTS):
            topics.add("yesno_generic")
        else:
            topics.add("generic")
    return topics


def topics_compatible(a: set[str], b: set[str]) -> bool:
    if a & b:
        return True
    generic_topics = {"generic", "time_generic", "yesno_generic", "money"}
    if a <= generic_topics and b <= generic_topics:
        return True
    return False


def semantic_question_key(question: str) -> str | None:
    """Conservative aliasing for short policy questions that embeddings often split."""
    topics = question_topics(question)
    if topics & {"international_delivery_time", "delivery_schedule", "damaged_goods", "logistics_issue"}:
        return None
    if any(token in question for token in ("经济", "最快", "加急", "特殊", "指定", "海外", "跨境", "国际")):
        return None

    normalized = normalize_for_dedup(question)
    normalized = re.sub(r"^(你好|您好|请问|麻烦问下|麻烦问一下|我想问下|我想问一下)", "", normalized)
    compact = normalized
    for token in ("我的", "你们的", "商品", "货品", "订单", "包裹", "快递", "物流", "一般", "预计", "大概", "通常"):
        compact = compact.replace(token, "")
    for source, target in (
        ("多长时间", "多久"),
        ("什么时候", "多久"),
        ("何时", "多久"),
        ("几天", "多久"),
        ("可以", ""),
        ("能不能", ""),
        ("能否", ""),
        ("能", ""),
        ("会", ""),
        ("可", ""),
        ("送达", "到"),
        ("送到", "到"),
        ("收到", "到"),
        ("到货", "到"),
        ("发出", "发货"),
        ("寄出", "发货"),
        ("出库", "发货"),
    ):
        compact = compact.replace(source, target)

    compact = re.sub(r"(的|吗|呢|啊|呀|吧|大约|左右|时间|需要|需要多久|多快|多久多久)+", "", compact)

    if "delivery_time" in topics and compact in {"多久到"}:
        return "semantic:delivery_time:多久到"
    if "ship_time" in topics and compact in {"多久发货"}:
        return "semantic:ship_time:多久发货"
    return None


def build_neighbor_graph(records: list[QARecord], embeddings: Any, embedding_name: str, args: argparse.Namespace) -> tuple[list[list[tuple[int, float]]], str, float]:
    threshold = args.similarity_threshold if args.similarity_threshold >= 0 else default_similarity_threshold(embedding_name)
    ann_backend = args.ann_backend
    n_samples = embeddings.shape[0]
    k = min(args.neighbors_k, n_samples)
    neighbors: list[list[tuple[int, float]]] = [[] for _ in range(n_samples)]
    topics_by_idx = [question_topics(record.question) for record in records]

    use_faiss = False
    if ann_backend in ("auto", "faiss") and embedding_name != "tfidf-char-2-4":
        try:
            import faiss

            use_faiss = True
            vectors = embeddings.astype(np.float32)
            cpu_index = faiss.IndexFlatIP(vectors.shape[1])
            faiss_backend_name = "faiss-cpu"
            if resolve_device(args.device) == "cuda" and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                    faiss_backend_name = "faiss-gpu"
                except Exception as exc:  # noqa: BLE001
                    print(f"FAISS GPU fallback to CPU. reason={exc}")
                    index = cpu_index
            else:
                index = cpu_index
            index.add(vectors)
            scores, indices = index.search(vectors, k)
            for i in range(n_samples):
                for j, score in zip(indices[i], scores[i]):
                    if i == j or j < 0:
                        continue
                    if float(score) >= threshold and topics_compatible(topics_by_idx[i], topics_by_idx[int(j)]):
                        neighbors[i].append((int(j), float(score)))
            return neighbors, faiss_backend_name, threshold
        except Exception as exc:  # noqa: BLE001
            if ann_backend == "faiss":
                raise RuntimeError(f"FAISS requested but unavailable: {exc}") from exc
            print(f"ANN fallback: faiss unavailable, use sklearn. reason={exc}")

    metric = "cosine"
    nn = NearestNeighbors(n_neighbors=k, metric=metric, algorithm="auto")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    for i in range(n_samples):
        for j, distance in zip(indices[i], distances[i]):
            if i == j:
                continue
            score = 1.0 - float(distance)
            if score >= threshold and topics_compatible(topics_by_idx[i], topics_by_idx[int(j)]):
                neighbors[i].append((int(j), score))
    return neighbors, "sklearn", threshold


def cluster_questions(records: list[QARecord], neighbors: list[list[tuple[int, float]]]) -> list[list[int]]:
    uf = UnionFind(len(records))
    exact_question_groups: dict[str, list[int]] = defaultdict(list)
    semantic_question_groups: dict[str, list[int]] = defaultdict(list)

    for i, record in enumerate(records):
        exact_question_groups[record.question_norm].append(i)
        semantic_key = semantic_question_key(record.question)
        if semantic_key:
            semantic_question_groups[semantic_key].append(i)
        for j, _score in neighbors[i]:
            uf.union(i, j)

    for indices in exact_question_groups.values():
        if len(indices) > 1:
            root = indices[0]
            for idx in indices[1:]:
                uf.union(root, idx)

    for indices in semantic_question_groups.values():
        if len(indices) > 1:
            root = indices[0]
            for idx in indices[1:]:
                uf.union(root, idx)

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(records)):
        clusters[uf.find(idx)].append(idx)

    return sorted(clusters.values(), key=len, reverse=True)


def classify_question_type(question: str) -> str:
    topics = question_topics(question)
    priority = [
        "invoice",
        "cod",
        "custom",
        "contact",
        "service_hours",
        "logistics_query",
        "damaged_goods",
        "logistics_issue",
        "pickup",
        "delivery_schedule",
        "international_delivery_time",
        "ship_time",
        "delivery_time",
        "return",
        "aftersales",
        "warranty",
        "stock",
        "payment",
        "address_change",
        "money",
        "time_generic",
        "yesno_generic",
        "generic",
    ]
    for topic in priority:
        if topic in topics:
            return topic
    return "generic"


def classify_stance(answer: str) -> str:
    if "无理由" in answer:
        answer = answer.replace("无理由", "可退货")
    for pattern in NEGATIVE_PATTERNS:
        if pattern in answer:
            return "negative"
    for pattern in POSITIVE_PATTERNS:
        if pattern in answer:
            return "positive"
    return "unknown"


def extract_time_ranges(answer: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for start, end, unit, suffix in WORKDAY_RE.findall(answer):
        low = float(start)
        high = float(end or start)
        if "小时" in unit or unit == "h":
            factor = 1.0
        else:
            factor = 24.0
        if suffix in {"以内", "之内", "内"} and not end:
            low = 0.0
        ranges.append((low * factor, high * factor))
    return ranges


def format_hours_for_policy(hours: float) -> str:
    if hours <= 0:
        return ""
    if hours % 24 == 0:
        days = int(hours // 24)
        return f"{days}个工作日"
    if hours < 24:
        return f"{int(hours)}小时"
    days = hours / 24
    if days.is_integer():
        return f"{int(days)}个工作日"
    return f"{days:.1f}个工作日"


def round_policy_hours(hours: float, *, ceil_value: bool) -> float:
    if hours <= 0:
        return 0.0
    if hours >= 24:
        days = hours / 24
        rounded_days = math.ceil(days) if ceil_value else max(1, math.floor(days))
        return float(rounded_days * 24)
    return float(math.ceil(hours) if ceil_value else max(1, math.floor(hours)))


def build_time_policy_answer(question_type: str, families: list[AnswerFamily], upper_percentile: float) -> str | None:
    if question_type == "international_delivery_time":
        return (
            "国际或跨境配送时效会受目的地国家和地区、清关进度、承运商线路、节假日等因素影响。 "
            "请以下单页展示的预计时效和物流追踪信息为准;如需确认具体国家或地区的配送时间,建议下单前联系客服核实。"
        )
    if question_type in TIME_POLICY_WINDOWS:
        low_text, high_text = TIME_POLICY_WINDOWS[question_type]
        window = f"{low_text}至{high_text}" if low_text != high_text else high_text
        if question_type == "ship_time":
            return (
                f"一般情况下,订单支付完成后我们会尽快处理,通常会在{window}内安排发货。 "
                "具体时效可能受库存、活动高峰、节假日及仓库处理进度影响,请以订单页和物流信息为准;如有加急需求,建议下单前联系客服确认。"
            )
        if question_type == "delivery_time":
            return (
                f"一般情况下,商品发出后通常会在{window}内送达。 "
                "具体到货时间会受收货地区、快递线路、天气和节假日影响,请以物流追踪信息为准;如需更准确时效,可以联系客服提供收货地区后确认。"
            )

    ranges = [time_range for family in families for time_range in family.time_ranges]
    if not ranges:
        return None

    positive_lows = [low for low, _high in ranges if low > 0]
    lows = np.asarray(positive_lows, dtype=np.float32)
    highs = np.asarray([high for _low, high in ranges if high > 0], dtype=np.float32)
    if highs.size == 0:
        return None
    upper_percentile = min(max(upper_percentile, 0.5), 1.0)
    low = float(np.quantile(lows, 0.15)) if lows.size else 0.0
    high = float(np.quantile(highs, upper_percentile))
    high = max(high, low)
    low = round_policy_hours(low, ceil_value=False)
    high = round_policy_hours(high, ceil_value=True)
    low_text = format_hours_for_policy(low)
    high_text = format_hours_for_policy(high)

    if low_text and high_text and low_text != high_text:
        window = f"{low_text}至{high_text}"
    else:
        window = high_text
    if not window:
        return None

    if question_type == "ship_time":
        return (
            f"一般情况下,订单支付完成后我们会尽快处理,通常会在{window}内安排发货。 "
            "具体时效可能受库存、活动高峰、节假日及仓库处理进度影响,请以订单页和物流信息为准;如有加急需求,建议下单前联系客服确认。"
        )
    if question_type == "delivery_time":
        return (
            f"一般情况下,商品发出后通常会在{window}内送达。 "
            "具体到货时间会受收货地区、快递线路、天气和节假日影响,请以物流追踪信息为准;如需更准确时效,可以联系客服提供收货地区后确认。"
        )
    return (
        f"一般情况下,相关处理或配送时效通常为{window}。 "
        "具体时间会受订单状态、商品库存、所在地区和物流情况影响,请以订单页、商品详情页或客服确认为准。"
    )


def build_time_policy_family(question_type: str, families: list[AnswerFamily], upper_percentile: float) -> AnswerFamily | None:
    if question_type not in TIME_POLICY_TYPES:
        return None
    policy_answer = build_time_policy_answer(question_type, families, upper_percentile)
    if not policy_answer:
        return None
    dominant = families[0]
    members = [member for family in families for member in family.members]
    representative = QARecord(
        idx=dominant.representative.idx,
        source_id=dominant.representative.source_id,
        question=dominant.representative.question,
        answer=policy_answer,
        question_norm=dominant.representative.question_norm,
        answer_norm=normalize_for_dedup(policy_answer),
    )
    return AnswerFamily(
        family_id=dominant.family_id,
        representative=representative,
        members=members,
        support_count=sum(family.support_count for family in families),
        quality_score=max(family.quality_score for family in families) + math.log2(len(families) + 1) * 0.2,
        stance="unknown",
        time_ranges=[time_range for family in families for time_range in family.time_ranges],
    )


def build_conservative_policy_answer(question_type: str) -> str | None:
    if question_type == "pickup":
        return (
            "自提服务会因商品、仓库和所在地区而不同,请以下单页可选配送方式为准。 "
            "如果页面支持自提,您可以按提示选择自提点并确认取货时间;如果页面没有自提选项,则默认通过快递配送。 如需确认具体商品是否可自提,建议下单前联系客服。"
        )
    if question_type == "damaged_goods":
        return (
            "收到商品如有破损、少件或异常,请先保留外包装、商品照片和物流面单等凭证,并尽快联系客服核实处理。 "
            "客服会根据实际情况协助您申请补发、换货、退款或联系物流核查,请不要自行丢弃相关凭证。"
        )
    return None


def build_conservative_policy_family(question_type: str, families: list[AnswerFamily]) -> AnswerFamily | None:
    if question_type not in CONSERVATIVE_POLICY_TYPES or not families:
        return None
    policy_answer = build_conservative_policy_answer(question_type)
    if not policy_answer:
        return None
    dominant = families[0]
    members = [member for family in families for member in family.members]
    representative = QARecord(
        idx=dominant.representative.idx,
        source_id=dominant.representative.source_id,
        question=dominant.representative.question,
        answer=policy_answer,
        question_norm=dominant.representative.question_norm,
        answer_norm=normalize_for_dedup(policy_answer),
    )
    return AnswerFamily(
        family_id=dominant.family_id,
        representative=representative,
        members=members,
        support_count=sum(family.support_count for family in families),
        quality_score=max(family.quality_score for family in families) + math.log2(len(families) + 1) * 0.2,
        stance="conditional",
        time_ranges=[],
    )


def answer_quality_score(record: QARecord, question_type: str) -> float:
    text = record.answer
    length = len(text)
    length_score = 1.0 - min(abs(length - 60) / 80.0, 1.0)
    filler_penalty = sum(text.count(term) for term in FILLER_TERMS) * 0.12
    useful_bonus = sum(1 for term in QUALITY_POSITIVE_TERMS if term in text) * 0.08
    digit_bonus = min(len(DIGIT_RE.findall(text)), 3) * 0.05
    risky_penalty = sum(1 for term in RISKY_PROMISE_TERMS if term in text) * 0.25
    punctuation_penalty = 0.15 if text.count("!") + text.count("~") > 3 else 0.0
    type_bonus = 0.0
    if question_type in {"ship_time", "delivery_time", "service_hours", "money", "return", "warranty"} and (digit_bonus > 0 or "具体" in text or "请" in text):
        type_bonus += 0.15
    if question_type in {"invoice", "cod", "custom", "stock", "address_change", "yesno_generic"} and classify_stance(text) != "unknown":
        type_bonus += 0.15
    return round(length_score + useful_bonus + digit_bonus + type_bonus - filler_penalty - risky_penalty - punctuation_penalty, 4)


def merge_answer_families(records: list[QARecord], question_type: str, threshold: float) -> list[AnswerFamily]:
    families: list[AnswerFamily] = []
    for record in sorted(records, key=lambda item: len(item.answer), reverse=True):
        placed = False
        for family in families:
            same_stance = classify_stance(record.answer) == family.stance or family.stance == "unknown"
            if not same_stance:
                continue
            left_len = len(record.answer_norm)
            right_len = len(family.representative.answer_norm)
            max_possible_ratio = 2 * min(left_len, right_len) / max(left_len + right_len, 1)
            if max_possible_ratio < threshold:
                continue
            sim = sequence_ratio(record.answer_norm, family.representative.answer_norm)
            if sim >= threshold and same_stance:
                family.members.append(record)
                family.support_count += 1
                placed = True
                break
        if placed:
            continue
        families.append(
            AnswerFamily(
                family_id=len(families),
                representative=record,
                members=[record],
                support_count=1,
                quality_score=answer_quality_score(record, question_type),
                stance=classify_stance(record.answer),
                time_ranges=extract_time_ranges(record.answer),
            )
        )

    for family in families:
        family.members.sort(key=lambda item: answer_quality_score(item, question_type), reverse=True)
        family.representative = family.members[0]
        family.quality_score = answer_quality_score(family.representative, question_type) + math.log2(family.support_count + 1) * 0.35
        family.stance = classify_stance(family.representative.answer)
        family.time_ranges = extract_time_ranges(family.representative.answer)

    families.sort(key=lambda item: (item.support_count, item.quality_score, len(item.representative.answer)), reverse=True)
    return families


def build_singleton_answer_families(records: list[QARecord], question_type: str) -> list[AnswerFamily]:
    families = [
        AnswerFamily(
            family_id=idx,
            representative=record,
            members=[record],
            support_count=1,
            quality_score=answer_quality_score(record, question_type),
            stance=classify_stance(record.answer),
            time_ranges=extract_time_ranges(record.answer),
        )
        for idx, record in enumerate(records)
    ]
    families.sort(key=lambda item: (item.support_count, item.quality_score, len(item.representative.answer)), reverse=True)
    return families


def ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def detect_contradictions(question_type: str, families: list[AnswerFamily]) -> tuple[list[AnswerFamily], list[AnswerFamily]]:
    if not families:
        return [], []

    kept: list[AnswerFamily] = []
    dropped: list[AnswerFamily] = []
    dominant = families[0]
    if dominant.support_count < 2:
        return list(families), []

    for family in families:
        reason: str | None = None
        if family.family_id == dominant.family_id:
            kept.append(family)
            continue

        if question_type == "yesno":
            if dominant.stance in {"positive", "negative"} and family.stance in {"positive", "negative"} and family.stance != dominant.stance:
                reason = f"opposite_stance_to_dominant:{dominant.stance}->{family.stance}"

        elif question_type in {"ship_time", "delivery_time", "time_generic"}:
            if dominant.time_ranges and family.time_ranges:
                dominant_overlap = any(ranges_overlap(a, b) for a in dominant.time_ranges for b in family.time_ranges)
                if not dominant_overlap:
                    reason = "non_overlapping_time_ranges"

        elif question_type in {"invoice", "cod", "custom", "stock", "address_change", "yesno_generic"}:
            dominant_conditional = any(term in dominant.representative.answer for term in CONDITIONAL_TERMS)
            family_conditional = any(term in family.representative.answer for term in CONDITIONAL_TERMS)
            if (
                not dominant_conditional
                and not family_conditional
                and dominant.stance in {"positive", "negative"}
                and family.stance in {"positive", "negative"}
                and family.stance != dominant.stance
            ):
                reason = f"policy_conflict:{dominant.stance}->{family.stance}"

        if reason and family.support_count <= dominant.support_count:
            family.contradiction_reason = reason
            dropped.append(family)
        else:
            kept.append(family)

    return kept, dropped


def choose_canonical_question(cluster_records: list[QARecord]) -> QARecord:
    counts = Counter(record.question_norm for record in cluster_records)
    ranked = sorted(
        cluster_records,
        key=lambda item: (counts[item.question_norm], -abs(len(item.question) - 12), len(item.question)),
        reverse=True,
    )
    return ranked[0]


def select_question_variants(canonical: QARecord, family: AnswerFamily, limit: int) -> list[QARecord]:
    limit = max(1, limit)
    variants: list[QARecord] = [canonical]
    seen = {canonical.question_norm}
    ranked_members = sorted(family.members, key=lambda item: (item.question_norm == canonical.question_norm, -len(item.question)), reverse=True)
    for member in ranked_members:
        if member.question_norm in seen:
            continue
        variants.append(member)
        seen.add(member.question_norm)
        if len(variants) >= limit:
            break
    return variants


def build_cluster_outputs(records: list[QARecord], clusters: list[list[int]], args: argparse.Namespace) -> tuple[list[dict], dict[str, list[dict]], dict[str, Any]]:
    audit = {
        "clusters_preview": [],
        "cluster_sizes": [],
        "dropped_contradictions": [],
        "dropped_low_rank_answers": [],
        "time_policy_consolidated": [],
        "policy_consolidated": [],
        "kept_answers": [],
    }
    output_records: list[dict] = []
    stats = Counter()

    for cluster_id, cluster_indices in enumerate(clusters):
        cluster_records = [records[idx] for idx in cluster_indices]
        canonical = choose_canonical_question(cluster_records)
        question_type = classify_question_type(canonical.question)
        if (
            (args.unify_time_answers and question_type in TIME_POLICY_TYPES)
            or question_type in CONSERVATIVE_POLICY_TYPES
            or len(cluster_records) >= LARGE_CLUSTER_FAST_FAMILY_THRESHOLD
        ):
            families = build_singleton_answer_families(cluster_records, question_type)
        else:
            families = merge_answer_families(cluster_records, question_type, args.answer_family_threshold)
        families = [family for family in families if family.support_count >= args.min_family_support]
        kept_families, dropped_contradictions = detect_contradictions(question_type, families)
        kept_families.sort(key=lambda item: (item.support_count, item.quality_score), reverse=True)
        conservative_policy_family = build_conservative_policy_family(question_type, kept_families)
        time_policy_family = (
            build_time_policy_family(question_type, kept_families, args.time_policy_upper_percentile)
            if args.unify_time_answers and not conservative_policy_family
            else None
        )
        if conservative_policy_family or time_policy_family:
            final_families = [conservative_policy_family or time_policy_family]
            low_rank_dropped = []
            audit_key = "policy_consolidated" if conservative_policy_family else "time_policy_consolidated"
            reason = "consolidated_into_conservative_policy" if conservative_policy_family else "consolidated_into_conservative_time_policy"
            for family in kept_families:
                audit[audit_key].append(
                    {
                        "cluster_id": cluster_id,
                        "canonical_question": canonical.question,
                        "answer": family.representative.answer,
                        "support_count": family.support_count,
                        "quality_score": round(family.quality_score, 4),
                        "reason": reason,
                    }
                )
        else:
            final_families = kept_families[: args.max_answers_per_cluster]
            low_rank_dropped = kept_families[args.max_answers_per_cluster :]
        final_question_variants = {
            family.family_id: select_question_variants(canonical, family, args.max_question_variants_per_answer)
            for family in final_families
        }

        stats["clusters_total"] += 1
        stats["raw_records_total"] += len(cluster_records)
        stats["families_total"] += len(families)
        stats["kept_records_total"] += sum(len(variants) for variants in final_question_variants.values())
        stats["contradictions_dropped_total"] += len(dropped_contradictions)
        stats["low_rank_dropped_total"] += len(low_rank_dropped)
        if len(cluster_records) > 1:
            stats["multi_record_clusters"] += 1

        preview = {
            "cluster_id": cluster_id,
            "cluster_size": len(cluster_records),
            "question_type": question_type,
            "canonical_question": canonical.question,
            "question_variants": list(dict.fromkeys(record.question for record in cluster_records))[:10],
            "kept_answers": [
                {
                    "support_count": family.support_count,
                    "quality_score": round(family.quality_score, 4),
                    "stance": family.stance,
                    "answer": family.representative.answer,
                }
                for family in final_families
            ],
        }
        if len(audit["clusters_preview"]) < 200:
            audit["clusters_preview"].append(preview)
        audit["cluster_sizes"].append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(cluster_records),
                "question_type": question_type,
                "canonical_question": canonical.question,
                "unique_questions": len({record.question_norm for record in cluster_records}),
                "answer_families": len(families),
                "kept_answers": len(final_families),
                "kept_records": sum(len(variants) for variants in final_question_variants.values()),
            }
        )

        for family in dropped_contradictions:
            audit["dropped_contradictions"].append(
                {
                    "cluster_id": cluster_id,
                    "canonical_question": canonical.question,
                    "answer": family.representative.answer,
                    "support_count": family.support_count,
                    "quality_score": round(family.quality_score, 4),
                    "reason": family.contradiction_reason,
                }
            )

        for family in low_rank_dropped:
            audit["dropped_low_rank_answers"].append(
                {
                    "cluster_id": cluster_id,
                    "canonical_question": canonical.question,
                    "answer": family.representative.answer,
                    "support_count": family.support_count,
                    "quality_score": round(family.quality_score, 4),
                    "reason": "ranked_below_cluster_cutoff",
                }
            )

        for family in final_families:
            for variant_idx, question_variant in enumerate(final_question_variants[family.family_id]):
                output_records.append(
                    {
                        "id": f"cluster_{cluster_id:05d}_ans_{family.family_id:02d}_q_{variant_idx:02d}",
                        "cluster_id": cluster_id,
                        "conversations": [
                            {"from": "user", "value": question_variant.question},
                            {"from": "assistant", "value": family.representative.answer},
                        ],
                        "metadata": {
                            "canonical_question": canonical.question,
                            "question_type": question_type,
                            "source_question": question_variant.question,
                            "source_ids": [member.source_id for member in family.members[:20]],
                            "support_count": family.support_count,
                            "quality_score": round(family.quality_score, 4),
                            "stance": family.stance,
                        },
                    }
                )
            audit["kept_answers"].append(
                {
                    "cluster_id": cluster_id,
                    "canonical_question": canonical.question,
                    "question_variants": [variant.question for variant in final_question_variants[family.family_id]],
                    "answer": family.representative.answer,
                    "support_count": family.support_count,
                    "quality_score": round(family.quality_score, 4),
                }
            )

    report = {
        "clusters_total": stats["clusters_total"],
        "multi_record_clusters": stats["multi_record_clusters"],
        "raw_records_total": stats["raw_records_total"],
        "answer_families_total": stats["families_total"],
        "kept_records_total": stats["kept_records_total"],
        "contradictions_dropped_total": stats["contradictions_dropped_total"],
        "low_rank_dropped_total": stats["low_rank_dropped_total"],
    }
    return output_records, audit, report


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    val_size = max(1, int(len(records) * val_ratio)) if records else 0
    val_set = set(indices[:val_size].tolist())
    train_records = [record for idx, record in enumerate(records) if idx not in val_set]
    val_records = [record for idx, record in enumerate(records) if idx in val_set]
    return train_records, val_records


AUGMENT_SYSTEM_PROMPT = """
你是电商客服SFT数据改写专家。请把FAQ改写为真实、多轮、可用于大模型客服SFT训练的高质量监督对话数据。

【核心目标】
生成的数据必须训练模型：

* 多轮上下文承接
* 问题闭环
* 客服决策能力
* 自动解决能力
* 高信息密度
* 强动作导向
* 强状态推进能力

【用户】

* 必须口语化
* 可包含疑惑、催促、不满
* 不要FAQ书面语

【assistant 核心原则】
assistant 每一轮回复必须至少完成以下之一：

* 推进问题处理状态
* 提供新的业务信息
* 给出明确操作动作
* 给出下一步处理方案
* 给出条件判断结果

禁止只进行情绪安抚。

【assistant 回复优先级】
assistant 回复必须优先包含：

1. 明确结论
2. 操作路径
3. 下一步动作
4. 必要时再进行解释

【动作导向要求】
assistant 回复必须至少包含以下之一：

* 操作入口
* 处理动作
* 状态更新
* 条件判断
* 下一步建议

【目标】
目标是：
让用户在当前对话中尽可能完成问题解决，
减少继续追问和人工介入。

【禁止】

* 不要输出“请稍等”“帮您查询”
* 不要输出占位符
* 不要编造FAQ中没有的时间/金额/政策承诺
* 不要索要好评
* 禁止无意义安抚
* 禁止重复用户问题
* 禁止生成泛客服回复

【禁止生成以下低质量回复】

* 您放心
* 好的呢
* 请稍等
* 我帮您查询
* 不是一对一
* 联系客服处理
* 我们会协助处理
* 稍后回复您
* 耐心等待

【高质量 assistant 回复示例】
✔ 您可以在【我的订单→申请售后】提交退款申请
✔ 如果订单未发货，可以直接申请取消
✔ 商品到站后会优先安排配送
✔ 如果页面无法修改，可以提供订单号我帮您跟进

【低质量 assistant 回复示例】
✘ 请稍等
✘ 我们会处理
✘ 您放心
✘ 联系客服

【对话长度】
2~4轮，自然即可。

【输出】
严格JSON格式，只输出：
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
"""

FORBIDDEN_AUGMENT_TERMS = ("请稍等", "稍等", "帮您查询", "好评", "占位符", "XXX", "xxx", "{{", "}}", "<", ">")


def build_augment_prompt(question: str, answer: str) -> str:
    return f"FAQ：\n问题：{question}\n答案：{answer}"


def augmentation_cache_key(question: str, answer: str, model: str) -> str:
    raw = json.dumps({"question": question, "answer": answer, "model": model}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_augmentation_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            if not line.strip():
                continue
            obj = json.loads(line)
            key = obj.get("cache_key")
            if isinstance(key, str) and isinstance(obj.get("dialogue"), dict):
                cache[key] = obj["dialogue"]
    return cache


def append_augmentation_cache(path: Path, cache_key: str, dialogue: dict, source_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"cache_key": cache_key, "source_id": source_id, "dialogue": dialogue}
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_original_qa(record: dict) -> tuple[str, str]:
    conversations = record.get("conversations", [])
    question = conversations[0].get("value", "") if len(conversations) >= 1 else ""
    answer = conversations[1].get("value", "") if len(conversations) >= 2 else ""
    return str(question), str(answer)


def validate_augmented_dialogue(dialogue: Any) -> list[dict[str, str]]:
    if not isinstance(dialogue, dict):
        raise ValueError("response is not a JSON object")
    messages = dialogue.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    if not 4 <= len(messages) <= 8 or len(messages) % 2 != 0:
        raise ValueError("dialogue must contain 2 to 4 user-assistant turns")

    normalized: list[dict[str, str]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        expected_role = "user" if idx % 2 == 0 else "assistant"
        role = message.get("role")
        content = clean_text(str(message.get("content", "")))
        if role != expected_role:
            raise ValueError(f"message {idx} role must be {expected_role}")
        if not content:
            raise ValueError(f"message {idx} content is empty")
        if any(term in content for term in FORBIDDEN_AUGMENT_TERMS):
            raise ValueError(f"message {idx} contains forbidden terms")
        normalized.append({"role": expected_role, "content": content})
    return normalized


def openai_chat_json(prompt: str, args: argparse.Namespace) -> dict:
    api_key = os.environ.get(args.augment_api_key_env)
    if not api_key:
        raise RuntimeError("OpenAI API key is not set")
    if not args.augment_api_base:
        raise RuntimeError("OpenAI API base URL is not set")

    schema = {
        "name": "faq_dialogue",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "messages": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "role": {"type": "string", "enum": ["user", "assistant"]},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
                }
            },
            "required": ["messages"],
        },
    }
    messages = [
        {"role": "system", "content": AUGMENT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    body = {
        "model": args.augment_model,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": schema},
        "temperature": 0.6,
    }
    try:
        return post_openai_chat_json(body, args)
    except Exception:  # noqa: BLE001
        fallback_body = {
            "model": args.augment_model,
            "messages": messages,
            "temperature": 0.4,
        }
        return post_openai_chat_json(fallback_body, args)


def parse_json_content(content: str) -> dict:
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


def post_openai_chat_json(body: dict, args: argparse.Namespace) -> dict:
    api_key = os.environ.get(args.augment_api_key_env)
    if not api_key:
        raise RuntimeError("OpenAI API key is not set")
    if not args.augment_api_base:
        raise RuntimeError("OpenAI API base URL is not set")
    url = args.augment_api_base.rstrip("/") + "/chat/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.augment_timeout) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return parse_json_content(content)


def choose_augmentation_indices(
    records: list[dict],
    ratio: float,
    seed: int,
    min_per_cluster: int = 0,
    strategy: str = "coverage",
) -> list[int]:
    ratio = min(max(ratio, 0.0), 1.0)
    if ratio <= 0:
        return []
    min_per_cluster = max(0, min_per_cluster)
    by_cluster: dict[Any, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        by_cluster[record.get("cluster_id")].append(idx)

    selected: list[int] = []
    rng = np.random.default_rng(seed)
    if strategy == "coverage":
        target_total = min(len(records), int(round(len(records) * ratio)))
        if target_total <= 0:
            return []
        selected_set: set[int] = set()
        cluster_ids = sorted(by_cluster, key=lambda item: (str(type(item)), str(item)))
        shuffled_cluster_ids = [cluster_ids[idx] for idx in rng.permutation(len(cluster_ids))]

        for cluster_id in shuffled_cluster_ids:
            if len(selected_set) >= target_total:
                break
            selected_set.add(int(rng.choice(by_cluster[cluster_id])))

        if len(selected_set) < target_total:
            remaining_by_cluster = {
                cluster_id: [idx for idx in indices if idx not in selected_set]
                for cluster_id, indices in by_cluster.items()
            }
            made_progress = True
            while len(selected_set) < target_total and made_progress:
                made_progress = False
                for cluster_id in shuffled_cluster_ids:
                    candidates = remaining_by_cluster[cluster_id]
                    if not candidates:
                        continue
                    chosen_pos = int(rng.integers(0, len(candidates)))
                    selected_set.add(candidates.pop(chosen_pos))
                    made_progress = True
                    if len(selected_set) >= target_total:
                        break
        return sorted(selected_set)

    for cluster_id in sorted(by_cluster, key=lambda item: (str(type(item)), str(item))):
        indices = by_cluster[cluster_id]
        sample_size = int(round(len(indices) * ratio))
        sample_size = min(len(indices), max(min_per_cluster, sample_size))
        if sample_size <= 0:
            continue
        selected.extend(rng.choice(indices, size=sample_size, replace=False).tolist())
    return sorted(selected)


def apply_dialogue_to_record(record: dict, messages: list[dict[str, str]]) -> dict:
    updated = dict(record)
    updated["conversations"] = [
        {"from": "user" if message["role"] == "user" else "assistant", "value": message["content"]}
        for message in messages
    ]
    metadata = dict(updated.get("metadata") or {})
    metadata["augmented_multi_turn"] = True
    metadata["augmentation_turns"] = len(messages) // 2
    updated["metadata"] = metadata
    return updated


def generate_augmented_dialogue(record: dict, args: argparse.Namespace) -> dict:
    question, answer = extract_original_qa(record)
    last_error: Exception | None = None
    for attempt in range(max(1, args.augment_max_retries)):
        try:
            dialogue = openai_chat_json(build_augment_prompt(question, answer), args)
            validate_augmented_dialogue(dialogue)
            return dialogue
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, KeyError, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < max(1, args.augment_max_retries):
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(str(last_error))


def augment_clustered_records(records: list[dict], args: argparse.Namespace, cache_path: Path) -> tuple[list[dict], dict[str, Any]]:
    selected_indices = choose_augmentation_indices(
        records,
        args.augment_ratio,
        args.seed,
        args.augment_min_per_cluster,
        args.augment_sampling_strategy,
    )
    if not selected_indices:
        return records, {"enabled": False, "selected": 0, "succeeded": 0, "failed": 0, "cache_hits": 0, "cache_path": str(cache_path)}

    cache = load_augmentation_cache(cache_path)
    output = list(records)
    stats = Counter({"selected": len(selected_indices)})
    missing_cache_keys = []
    for idx in selected_indices:
        question, answer = extract_original_qa(records[idx])
        key = augmentation_cache_key(question, answer, args.augment_model)
        if key not in cache:
            missing_cache_keys.append(key)
    api_key = os.environ.get(args.augment_api_key_env)
    if missing_cache_keys and not api_key:
        raise RuntimeError(f"OpenAI API key is not set, and {len(missing_cache_keys)} selected rows are missing from augmentation cache")

    pending: list[tuple[int, str]] = []
    for idx in selected_indices:
        record = records[idx]
        question, answer = extract_original_qa(record)
        key = augmentation_cache_key(question, answer, args.augment_model)
        dialogue = cache.get(key)
        if dialogue:
            stats["cache_hits"] += 1
            try:
                messages = validate_augmented_dialogue(dialogue)
                output[idx] = apply_dialogue_to_record(record, messages)
                stats["succeeded"] += 1
            except ValueError as exc:
                stats["failed"] += 1
                print(f"Augment cached validation failed id={record.get('id')} error={exc}")
            continue
        pending.append((idx, key))

    completed = stats["succeeded"] + stats["failed"]
    if completed and (completed % 50 == 0 or completed == len(selected_indices)):
        print(f"Augment progress: {completed}/{len(selected_indices)} selected, succeeded={stats['succeeded']}, failed={stats['failed']}, cache_hits={stats['cache_hits']}")

    concurrency = max(1, args.augment_concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_item = {
            executor.submit(generate_augmented_dialogue, records[idx], args): (idx, key)
            for idx, key in pending
        }
        for future in as_completed(future_to_item):
            idx, key = future_to_item[future]
            record = records[idx]
            try:
                dialogue = future.result()
                messages = validate_augmented_dialogue(dialogue)
                append_augmentation_cache(cache_path, key, dialogue, str(record.get("id", "")))
                cache[key] = dialogue
                output[idx] = apply_dialogue_to_record(record, messages)
                stats["succeeded"] += 1
            except Exception as exc:  # noqa: BLE001
                stats["failed"] += 1
                print(f"Augment failed id={record.get('id')} error={exc}")

            completed = stats["succeeded"] + stats["failed"]
            if completed % 50 == 0 or completed == len(selected_indices):
                print(f"Augment progress: {completed}/{len(selected_indices)} selected, succeeded={stats['succeeded']}, failed={stats['failed']}, cache_hits={stats['cache_hits']}")

    return output, {
        "enabled": True,
        "model": args.augment_model,
        "ratio": args.augment_ratio,
        "sampling_strategy": args.augment_sampling_strategy,
        "min_per_cluster": args.augment_min_per_cluster,
        "selected": stats["selected"],
        "succeeded": stats["succeeded"],
        "failed": stats["failed"],
        "cache_hits": stats["cache_hits"],
        "concurrency": concurrency,
        "cache_path": str(cache_path),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    audit_dir = Path(args.audit_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input)
    embeddings, embedding_name = build_embeddings(records, args)
    neighbors, ann_backend, sim_threshold = build_neighbor_graph(records, embeddings, embedding_name, args)
    clusters = cluster_questions(records, neighbors)
    output_records, audit, cluster_report = build_cluster_outputs(records, clusters, args)
    augment_cache_path = Path(args.augment_cache) if args.augment_cache else audit_dir / "faq_dialogue_augmentation_cache.jsonl"
    output_records, augmentation_report = augment_clustered_records(output_records, args, augment_cache_path)
    train_records, val_records = split_records(output_records, args.val_ratio, args.seed)

    append_jsonl(output_dir / "all.jsonl", output_records)
    append_jsonl(output_dir / "train.jsonl", train_records)
    append_jsonl(output_dir / "val.jsonl", val_records)
    (audit_dir / "clusters_preview.json").write_text(json.dumps(audit["clusters_preview"], ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(audit_dir / "cluster_sizes.jsonl", audit["cluster_sizes"])
    append_jsonl(audit_dir / "dropped_contradictions.jsonl", audit["dropped_contradictions"])
    append_jsonl(audit_dir / "dropped_low_rank_answers.jsonl", audit["dropped_low_rank_answers"])
    append_jsonl(audit_dir / "time_policy_consolidated.jsonl", audit["time_policy_consolidated"])
    append_jsonl(audit_dir / "policy_consolidated.jsonl", audit["policy_consolidated"])
    append_jsonl(audit_dir / "kept_answers.jsonl", audit["kept_answers"])

    report = {
        "input_path": str(Path(args.input).resolve()),
        "output_dir": str(output_dir.resolve()),
        "audit_dir": str(audit_dir.resolve()),
        "embedding_backend_used": embedding_name,
        "ann_backend_used": ann_backend,
        "device_requested": args.device,
        "device_resolved": resolve_device(args.device),
        "batch_size": args.batch_size,
        "question_similarity_threshold": sim_threshold,
        "neighbors_k": args.neighbors_k,
        "answer_family_threshold": args.answer_family_threshold,
        "max_answers_per_cluster": args.max_answers_per_cluster,
        "unify_time_answers": args.unify_time_answers,
        "time_policy_upper_percentile": args.time_policy_upper_percentile,
        "max_question_variants_per_answer": args.max_question_variants_per_answer,
        "val_ratio": args.val_ratio,
        "augment_ratio": args.augment_ratio,
        "augment_sampling_strategy": args.augment_sampling_strategy,
        "augment_min_per_cluster": args.augment_min_per_cluster,
        "augment_concurrency": args.augment_concurrency,
        "augmentation": augmentation_report,
        **cluster_report,
        "train_size": len(train_records),
        "val_size": len(val_records),
    }
    (audit_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
