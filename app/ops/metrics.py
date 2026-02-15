import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


CHAT_REQUESTS_TOTAL = Counter("chat_requests_total", "Total number of /chat requests")
CHAT_ERRORS_TOTAL = Counter("chat_errors_total", "Total number of failed /chat requests", ["type"])
FALLBACK_ANSWERS_TOTAL = Counter(
    "fallback_answers_total", "Total number of extractive fallback answers"
)
CACHE_HITS_TOTAL = Counter("cache_hits_total", "Total number of cache hits")
RERANK_FAILURES_TOTAL = Counter("rerank_failures_total", "Total number of rerank failures")

CHAT_LATENCY_SECONDS = Histogram("chat_latency_seconds", "End-to-end /chat latency in seconds")
RETRIEVE_LATENCY_SECONDS = Histogram(
    "retrieve_latency_seconds", "Retrieval stage latency in seconds"
)
RERANK_LATENCY_SECONDS = Histogram("rerank_latency_seconds", "Rerank stage latency in seconds")
GENERATE_LATENCY_SECONDS = Histogram(
    "generate_latency_seconds", "Generation stage latency in seconds"
)

QDRANT_COLLECTION_POINTS = Gauge(
    "qdrant_collection_points", "Estimated number of points in current Qdrant collection"
)
DOCS_INDEXED_TOTAL = Gauge("docs_indexed_total", "Number of docs indexed in last ingest run")


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
    CHAT_ERRORS_TOTAL.labels(type=error_type).inc()


def inc_fallback_answers() -> None:
    FALLBACK_ANSWERS_TOTAL.inc()


def inc_cache_hits() -> None:
    CACHE_HITS_TOTAL.inc()


def inc_rerank_failures() -> None:
    RERANK_FAILURES_TOTAL.inc()


def set_qdrant_collection_points(count: int) -> None:
    QDRANT_COLLECTION_POINTS.set(max(0, count))


def set_docs_indexed_total(count: int) -> None:
    DOCS_INDEXED_TOTAL.set(max(0, count))


def render_metrics() -> bytes:
    return generate_latest()


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
