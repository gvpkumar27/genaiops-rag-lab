"""Hybrid retrieval with semantic search plus lexical fallback."""

import re
from typing import Any

from qdrant_client import QdrantClient

from app.config import settings
from app.rag.embeddings import embed_text
from app.rag.query import normalized_question


def _merge_fragmented_letters(text: str) -> str:
    pattern = re.compile(r"\b(?:[A-Za-z]{1,2}\s+){3,}[A-Za-z]{1,12}\b")

    def repl(match: re.Match[str]) -> str:
        tokens = match.group(0).split()
        single_count = sum(1 for token in tokens if len(token) == 1)
        if single_count >= 2:
            return "".join(tokens)
        return match.group(0)

    out = text
    for _ in range(2):
        newer = pattern.sub(repl, out)
        if newer == out:
            break
        out = newer
    return out


def _tokenize(text: str) -> set[str]:
    text = _merge_fragmented_letters(text)
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _contamination_penalty(text: str) -> float:
    if not settings.ENABLE_CONTAMINATION_FILTER:
        return 0.0

    low = text.lower()
    markers = (
        "article tags",
        "last updated",
        "sign in",
        "follow us",
        "cookie policy",
        "advertisement",
        "all rights reserved",
    )
    hits = sum(1 for marker in markers if marker in low)
    return min(0.45, hits * 0.12)


def _make_hit(
    source: str,
    chunk_id: int,
    text: str,
    rank_score: float,
    semantic_score: float,
) -> dict[str, Any]:
    return {
        "score": semantic_score,
        "rank_score": rank_score,
        "source": source,
        "chunk_id": chunk_id,
        "text": text,
    }


def _score_semantic_result(
    question_tokens: set[str],
    text: str,
    semantic_score: float,
) -> float:
    text_tokens = _tokenize(text)
    overlap = len(question_tokens & text_tokens)
    keyword_score = overlap / max(1, len(question_tokens))
    return semantic_score + (0.25 * keyword_score) - _contamination_penalty(text)


def _insert_or_update(by_key: dict, key: tuple[str, int], candidate: dict[str, Any]) -> None:
    current = by_key.get(key)
    if current is None or candidate["rank_score"] > current["rank_score"]:
        by_key[key] = candidate


def _semantic_candidates(
    client: QdrantClient,
    query_vector: list[float],
    candidate_k: int,
    question_tokens: set[str],
) -> dict[tuple[str, int], dict[str, Any]]:
    semantic_results = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=candidate_k,
    )
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for result in semantic_results:
        payload = result.payload or {}
        source = payload.get("source", "unknown")
        chunk_id = int(payload.get("chunk_id", -1))
        text = payload.get("text", "")
        semantic_score = float(result.score)
        rank_score = _score_semantic_result(question_tokens, text, semantic_score)
        key = (source, chunk_id)
        _insert_or_update(
            by_key,
            key,
            _make_hit(source, chunk_id, text, rank_score, semantic_score),
        )
    return by_key


def _lexical_rank(question_tokens: set[str], text: str) -> float | None:
    text_tokens = _tokenize(text)
    overlap = len(question_tokens & text_tokens)
    if overlap == 0:
        return None
    keyword_score = overlap / max(1, len(question_tokens))
    return 0.15 + (0.55 * keyword_score) - _contamination_penalty(text)


def _apply_lexical_point(
    by_key: dict[tuple[str, int], dict[str, Any]],
    point: Any,
    question_tokens: set[str],
) -> None:
    payload = point.payload or {}
    source = payload.get("source", "unknown")
    chunk_id = int(payload.get("chunk_id", -1))
    text = payload.get("text", "")
    rank_score = _lexical_rank(question_tokens, text)
    if rank_score is None:
        return

    key = (source, chunk_id)
    current = by_key.get(key)
    semantic_score = current["score"] if current else 0.0
    best_rank = max(rank_score, current["rank_score"]) if current else rank_score
    _insert_or_update(
        by_key,
        key,
        _make_hit(source, chunk_id, text, best_rank, semantic_score),
    )


def _apply_lexical_fallback(
    client: QdrantClient,
    by_key: dict[tuple[str, int], dict[str, Any]],
    question_tokens: set[str],
) -> None:
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=offset,
        )
        if not points:
            break
        for point in points:
            _apply_lexical_point(by_key, point, question_tokens)
        if offset is None:
            break


def retrieve(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    client = QdrantClient(url=settings.QDRANT_URL)
    normalized = normalized_question(question)
    query_for_search = normalized or question
    query_vector = embed_text(query_for_search)
    question_tokens = _tokenize(query_for_search)

    final_k = top_k or settings.TOP_K
    candidate_k = max(final_k, settings.RETRIEVE_CANDIDATES, final_k * 4)

    by_key = _semantic_candidates(client, query_vector, candidate_k, question_tokens)
    _apply_lexical_fallback(client, by_key, question_tokens)

    hits = list(by_key.values())
    hits.sort(key=lambda hit: hit["rank_score"], reverse=True)
    return hits[:final_k]
