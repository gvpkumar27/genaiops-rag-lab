import time
from contextlib import contextmanager

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


CHAT_REQUESTS_TOTAL = Counter("queries_total", "Total number of query requests")
CHAT_ERRORS_TOTAL = Counter(
    "query_errors_total",
    "Total number of failed query requests",
    ["reason"],
)
FALLBACK_ANSWERS_TOTAL = Counter(
    "fallback_total",
    "Total number of fallback answers",
)
CACHE_HITS_TOTAL = Counter("cache_hits_total", "Total number of cache hits")
RERANK_FAILURES_TOTAL = Counter(
    "rank_failures_total",
    "Total number of ranking failures",
)
PROMPT_ATTACKS_BLOCKED_TOTAL = Counter(
    "policy_blocks_total",
    "Total number of policy-blocked requests",
    ["policy"],
)
OUT_OF_SCOPE_BLOCKS_TOTAL = Counter(
    "scope_blocks_total",
    "Total number of scope-blocked requests",
)

CHAT_LATENCY_SECONDS = Histogram("query_seconds", "End-to-end query latency in seconds")
RETRIEVE_LATENCY_SECONDS = Histogram(
    "search_seconds",
    "Search stage latency in seconds",
)
RERANK_LATENCY_SECONDS = Histogram("rank_seconds", "Ranking stage latency in seconds")
GENERATE_LATENCY_SECONDS = Histogram(
    "answer_seconds",
    "Answer stage latency in seconds",
)

QDRANT_COLLECTION_POINTS = Gauge(
    "index_points_total",
    "Estimated number of points in current index",
)
DOCS_INDEXED_TOTAL = Gauge(
    "docs_indexed_total",
    "Number of documents indexed in last ingest run",
)


@contextmanager
def timer():
    start = time.time()
    yield lambda: time.time() - start


def observe_chat_latency(sec: float) -> None:
    CHAT_LATENCY_SECONDS.observe(sec)


def observe_retrieve_latency(sec: float) -> None:
    RETRIEVE_LATENCY_SECONDS.observe(sec)


def observe_rerank_latency(sec: float) -> None:
    RERANK_LATENCY_SECONDS.observe(sec)


def observe_generate_latency(sec: float) -> None:
    GENERATE_LATENCY_SECONDS.observe(sec)


def inc_chat_requests() -> None:
    CHAT_REQUESTS_TOTAL.inc()


def inc_chat_error(error_type: str) -> None:
    CHAT_ERRORS_TOTAL.labels(reason=error_type).inc()


def inc_fallback_answers() -> None:
    FALLBACK_ANSWERS_TOTAL.inc()


def inc_cache_hits() -> None:
    CACHE_HITS_TOTAL.inc()


def inc_rerank_failures() -> None:
    RERANK_FAILURES_TOTAL.inc()


def inc_prompt_attack_blocked(category: str) -> None:
    PROMPT_ATTACKS_BLOCKED_TOTAL.labels(policy=category).inc()


def inc_out_of_scope_block() -> None:
    OUT_OF_SCOPE_BLOCKS_TOTAL.inc()


def set_qdrant_collection_points(count: int) -> None:
    QDRANT_COLLECTION_POINTS.set(max(0, count))


def set_docs_indexed_total(count: int) -> None:
    DOCS_INDEXED_TOTAL.set(max(0, count))


def render_metrics() -> bytes:
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
