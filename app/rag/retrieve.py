"""Hybrid retrieval with semantic search plus lexical fallback."""

import re
from typing import Any

from qdrant_client import QdrantClient

from app.config import settings
from app.rag.embeddings import embed_text
from app.rag.intent import detect_answer_intent, intent_signal_score, section_type_boost
from app.rag.query import normalized_question, rewrite_question_variants


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
    section_type: str,
    source_trust: float = 0.0,
    is_stale: bool = False,
) -> dict[str, Any]:
    return {
        "score": semantic_score,
        "rank_score": rank_score,
        "source": source,
        "chunk_id": chunk_id,
        "text": text,
        "section_type": section_type,
        "source_trust": source_trust,
        "is_stale": is_stale,
    }


def _source_quality_adjustment(source_trust: float, is_stale: bool) -> float:
    trust_boost = max(0.0, min(source_trust, 1.0)) * 0.08
    stale_penalty = 0.04 if is_stale else 0.0
    return trust_boost - stale_penalty


def _score_semantic_result(
    question_tokens: set[str],
    text: str,
    semantic_score: float,
    intent: str,
    section_type: str,
    source_trust: float = 0.0,
    is_stale: bool = False,
) -> float:
    text_tokens = _tokenize(text)
    overlap = len(question_tokens & text_tokens)
    keyword_score = overlap / max(1, len(question_tokens))
    return (
        semantic_score
        + (0.25 * keyword_score)
        + intent_signal_score(text, intent)
        + section_type_boost(section_type, intent)
        + _source_quality_adjustment(source_trust, is_stale)
        - _contamination_penalty(text)
    )


def _insert_or_update(by_key: dict, key: tuple[str, int], candidate: dict[str, Any]) -> None:
    current = by_key.get(key)
    if current is None or candidate["rank_score"] > current["rank_score"]:
        by_key[key] = candidate


def _semantic_candidates(
    client: QdrantClient,
    query_vectors: list[list[float]],
    candidate_k: int,
    question_tokens: set[str],
    intent: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for query_vector in query_vectors:
        semantic_results = client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=candidate_k,
        )
        for result in semantic_results:
            payload = result.payload or {}
            source = payload.get("source", "unknown")
            chunk_id = int(payload.get("chunk_id", -1))
            text = payload.get("text", "")
            section_type = str(payload.get("section_type", "general"))
            source_trust = float(payload.get("source_trust", 0.0))
            is_stale = bool(payload.get("is_stale", False))
            semantic_score = float(result.score)
            rank_score = _score_semantic_result(
                question_tokens,
                text,
                semantic_score,
                intent,
                section_type,
                source_trust,
                is_stale,
            )
            key = (source, chunk_id)
            _insert_or_update(
                by_key,
                key,
                _make_hit(
                    source,
                    chunk_id,
                    text,
                    rank_score,
                    semantic_score,
                    section_type,
                    source_trust,
                    is_stale,
                ),
            )
    return by_key


def _lexical_rank(
    question_tokens: set[str],
    text: str,
    intent: str,
    section_type: str = "general",
    source_trust: float = 0.0,
    is_stale: bool = False,
) -> float | None:
    text_tokens = _tokenize(text)
    overlap = len(question_tokens & text_tokens)
    if overlap == 0:
        return None
    keyword_score = overlap / max(1, len(question_tokens))
    return (
        0.15
        + (0.55 * keyword_score)
        + intent_signal_score(text, intent)
        + section_type_boost(section_type, intent)
        + _source_quality_adjustment(source_trust, is_stale)
        - _contamination_penalty(text)
    )


def _apply_lexical_point(
    by_key: dict[tuple[str, int], dict[str, Any]],
    point: Any,
    question_tokens: set[str],
    intent: str,
) -> None:
    payload = point.payload or {}
    source = payload.get("source", "unknown")
    chunk_id = int(payload.get("chunk_id", -1))
    text = payload.get("text", "")
    section_type = str(payload.get("section_type", "general"))
    source_trust = float(payload.get("source_trust", 0.0))
    is_stale = bool(payload.get("is_stale", False))
    rank_score = _lexical_rank(question_tokens, text, intent, section_type, source_trust, is_stale)
    if rank_score is None:
        return

    key = (source, chunk_id)
    current = by_key.get(key)
    semantic_score = current["score"] if current else 0.0
    best_rank = max(rank_score, current["rank_score"]) if current else rank_score
    _insert_or_update(
        by_key,
        key,
        _make_hit(
            source,
            chunk_id,
            text,
            best_rank,
            semantic_score,
            section_type,
            source_trust,
            is_stale,
        ),
    )


def _apply_lexical_fallback(
    client: QdrantClient,
    by_key: dict[tuple[str, int], dict[str, Any]],
    question_tokens: set[str],
    intent: str,
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
            _apply_lexical_point(by_key, point, question_tokens, intent)
        if offset is None:
            break


def _dedupe_hits_by_text(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        key = re.sub(r"\s+", " ", hit.get("text", "").strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(hit)
    return deduped


def retrieve(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    client = QdrantClient(url=settings.QDRANT_URL)
    normalized = normalized_question(question)
    query_for_search = normalized or question
    variants = rewrite_question_variants(question)
    if not settings.ENABLE_DUAL_QUERY_RETRIEVAL:
        variants = [query_for_search]
    query_vectors = [embed_text(variant) for variant in variants]
    question_tokens = set()
    for variant in variants:
        question_tokens |= _tokenize(variant)
    intent = detect_answer_intent(question)

    final_k = top_k or settings.TOP_K
    candidate_k = max(final_k, settings.RETRIEVE_CANDIDATES, final_k * 4)

    by_key = _semantic_candidates(
        client,
        query_vectors,
        candidate_k,
        question_tokens,
        intent,
    )
    _apply_lexical_fallback(client, by_key, question_tokens, intent)

    hits = _dedupe_hits_by_text(list(by_key.values()))
    hits.sort(key=lambda hit: hit["rank_score"], reverse=True)
    return hits[:final_k]
