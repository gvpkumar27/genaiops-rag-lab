"""FastAPI entrypoint for LocalDocChat RAG service."""
# pylint: disable=too-many-lines

import re
import time
import uuid
import hashlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import requests
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.ops.logging import log_event
from app.ops.metrics import (
    inc_cache_hits,
    inc_chat_error,
    inc_out_of_scope_block,
    inc_chat_requests,
    inc_fallback_answers,
    inc_prompt_attack_blocked,
    inc_rerank_failures,
    metrics_content_type,
    observe_chat_latency,
    observe_generate_latency,
    observe_rerank_latency,
    observe_retrieve_latency,
    render_metrics,
    set_docs_indexed_total,
    set_qdrant_collection_points,
    timer,
)
from app.rag.cache import get_cached, set_cached
from app.rag.generate import generate_answer
from app.rag.guardrails import analyze_question_risk, validate_guardrails_config
from app.rag.rerank import llm_rerank
from app.rag.retrieve import retrieve
from app.schemas import ChatRequest, ChatResponse, Citation


def _is_local_bind_host(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


def startup_security_check() -> None:
    if not settings.SYSTEM_PROMPT and not settings.SYSTEM_PROMPT_FILE:
        raise RuntimeError(
            "Missing prompt configuration. Set SYSTEM_PROMPT_FILE or SYSTEM_PROMPT "
            "in local .env."
        )
    if (
        settings.REQUIRE_API_KEY_ON_NON_LOCALHOST
        and not settings.API_KEY
        and not _is_local_bind_host(settings.API_HOST)
    ):
        raise RuntimeError(
            "Unsafe configuration: API_HOST is non-localhost but API_KEY is empty. "
            "Set API_KEY or set REQUIRE_API_KEY_ON_NON_LOCALHOST=false only for "
            "trusted local testing."
        )
    if settings.ENABLE_PROMPT_GUARDRAILS:
        if settings.REQUIRE_PRIVATE_GUARDRAILS and not settings.GUARDRAILS_FILE:
            raise RuntimeError(
                "Missing private guardrails configuration. Set GUARDRAILS_FILE "
                "when REQUIRE_PRIVATE_GUARDRAILS=true."
            )
        validate_guardrails_config()
    _refresh_runtime_gauges()


@asynccontextmanager
async def lifespan(_: FastAPI):
    startup_security_check()
    yield


app = FastAPI(title="LocalDocChat", lifespan=lifespan)


def require_api_key(request: Request) -> None:
    if (
        settings.REQUIRE_API_KEY_ON_NON_LOCALHOST
        and not settings.API_KEY
        and not _is_local_bind_host(settings.API_HOST)
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Server misconfiguration: API key required "
                "for non-localhost bind"
            ),
        )
    if not settings.API_KEY:
        return
    if request.headers.get("x-api-key", "") != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "what",
    "when",
    "where",
    "why",
    "how",
    "can",
    "does",
    "did",
    "will",
    "would",
    "should",
    "about",
    "based",
    "provided",
    "documents",
}

_UNKNOWN_SENTENCE = "I don't know based on the provided documents."


@dataclass
class ChatState:
    started: float
    retrieve_sec: float = 0.0
    rerank_sec: float = 0.0
    generate_sec: float = 0.0
    cache_hit: bool = False
    fallback_used: bool = False


def _tok(text: str) -> set[str]:
    toks = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    return {t for t in toks if t not in _STOPWORDS}


def _is_meaningful_question(question: str) -> bool:
    if not question.strip():
        return False
    return bool(re.search(r"[a-zA-Z0-9]{2,}", question))


def _question_text_overlap(question: str, text: str) -> float:
    q = _tok(question)
    if not q:
        return 0.0
    t = _tok(text)
    return len(q & t) / max(1, len(q))


def _question_context_coverage(question: str, hits: list[dict]) -> float:
    question_tokens = _tok(question)
    if not question_tokens:
        return 0.0
    context_tokens = set()
    for hit in hits:
        context_tokens |= _tok(hit.get("text", ""))
    if not context_tokens:
        return 0.0
    return len(question_tokens & context_tokens) / max(1, len(question_tokens))


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_summary_intent(question: str) -> bool:
    pattern = r"\b(summarize|summarise|summary|overview|highlights|features)\b"
    return bool(re.search(pattern, question.lower()))


def _is_cross_document_intent(question: str) -> bool:
    pattern = (
        r"\b("
        r"all\s+distinct\s+documents|"
        r"each\s+document|"
        r"across\s+documents|"
        r"across\s+all\s+documents|"
        r"which\s+document|"
        r"documents?\s+currently\s+indexed|"
        r"list\s+documents|"
        r"all\s+documents"
        r")\b"
    )
    return bool(re.search(pattern, question.lower()))


def _is_document_inventory_intent(question: str) -> bool:
    pattern = (
        r"\b("
        r"list\s+all\s+distinct\s+documents|"
        r"documents?\s+currently\s+indexed|"
        r"list\s+indexed\s+documents|"
        r"each\s+document.*summary|"
        r"summary\s+of\s+each\s+document"
        r")\b"
    )
    return bool(re.search(pattern, question.lower()))


def _is_good_fallback_sentence(sentence: str) -> bool:
    words = sentence.split()
    if len(words) < 6 or len(words) > 36:
        return False

    low = sentence.lower()
    noise_markers = (
        "article tags",
        "last updated",
        "sign in",
        "output",
        "driver code",
        "formatted current date and time",
    )
    if any(marker in low for marker in noise_markers):
        return False

    alpha_count = sum(ch.isalpha() for ch in sentence)
    ratio = alpha_count / max(1, len(sentence))
    return ratio >= 0.65


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _rank_fallback_candidates(
    question_tokens: set[str], hits: list[dict]
) -> list[tuple[int, int, str, str, int]]:
    ranked: list[tuple[int, int, str, str, int]] = []
    for hit in hits:
        text = _clean_text(hit.get("text", ""))
        source = hit.get("source", "unknown")
        chunk_id = int(hit.get("chunk_id", -1))
        for sentence in _split_sentences(text):
            sentence_tokens = _tok(sentence)
            overlap_count = len(question_tokens & sentence_tokens)
            token_coverage = overlap_count / max(1, len(question_tokens))
            sentence_word_count = len(sentence.split())
            if (
                overlap_count >= 2
                and token_coverage >= 0.3
                and sentence_word_count >= 6
                and _is_good_fallback_sentence(sentence)
            ):
                ranked.append(
                    (
                        overlap_count,
                        sentence_word_count,
                        sentence,
                        source,
                        chunk_id,
                    )
                )
    return ranked


def _pick_unique_sentences(
    candidates: list[tuple[str, str, int]], limit: int
) -> list[tuple[str, str, int]]:
    picked: list[tuple[str, str, int]] = []
    seen = set()
    for sentence, source, chunk_id in candidates:
        key = (source, chunk_id, sentence[:80])
        if key in seen:
            continue
        seen.add(key)
        picked.append((sentence, source, chunk_id))
        if len(picked) >= limit:
            break
    return picked


def _summary_fallback_candidates(hits: list[dict]) -> list[tuple[str, str, int]]:
    candidates: list[tuple[str, str, int]] = []
    for hit in hits[:5]:
        text = _clean_text(hit.get("text", ""))
        source = hit.get("source", "unknown")
        chunk_id = int(hit.get("chunk_id", -1))
        for sentence in _split_sentences(text):
            if _is_good_fallback_sentence(sentence):
                candidates.append((sentence, source, chunk_id))
    return candidates


def _format_fallback(
    prefix: str, picked: list[tuple[str, str, int]]
) -> str | None:
    if not picked:
        return None
    lines = [f"- {sent} [{source} | {chunk_id}]" for sent, source, chunk_id in picked]
    return f"{prefix}\n" + "\n".join(lines)


def _extractive_fallback(question: str, hits: list[dict]) -> str | None:
    question_tokens = _tok(question)
    if not question_tokens:
        return None

    ranked = _rank_fallback_candidates(question_tokens, hits)
    if ranked:
        ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
        candidates = [(sent, source, chunk_id) for _, _, sent, source, chunk_id in ranked]
        picked = _pick_unique_sentences(candidates, limit=3)
        return _format_fallback("Based on the uploaded documents:", picked)

    if not _is_summary_intent(question):
        return None

    summary_candidates = _summary_fallback_candidates(hits)
    picked = _pick_unique_sentences(summary_candidates, limit=5)
    return _format_fallback("Summary from the uploaded documents:", picked)


def _answer_groundedness(answer: str, hits: list[dict]) -> float:
    answer_tokens = _tok(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for hit in hits:
        context_tokens |= _tok(hit.get("text", ""))
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / max(1, len(answer_tokens))


def _has_inline_citation(answer: str) -> bool:
    return bool(re.search(r"\[[^\[\]\n]+\|\s*-?\d+\]", answer))


def _top_citation_tag(hits: list[dict]) -> str:
    if not hits:
        return "[unknown | -1]"
    hit = hits[0]
    source = hit.get("source", "unknown")
    chunk_id = int(hit.get("chunk_id", -1))
    return f"[{source} | {chunk_id}]"


def _ensure_inline_citation(answer: str, hits: list[dict]) -> str:
    if _has_inline_citation(answer):
        return answer
    output = answer.strip()
    if not output:
        return output
    return f"{output} {_top_citation_tag(hits)}"


def _contains_unknown_marker(answer: str) -> bool:
    return _UNKNOWN_SENTENCE.lower() in answer.strip().lower()


def _strip_unknown_marker(answer: str) -> str:
    out = re.sub(re.escape(_UNKNOWN_SENTENCE), "", answer, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip(" \n\r\t-")


def _public_source_labels(citations: list[Citation]) -> dict[str, str]:
    labels: dict[str, str] = {}
    next_number = 1
    for citation in citations:
        source = citation.source
        if source not in labels:
            labels[source] = f"Document {next_number}"
            next_number += 1
    return labels


def _sanitize_answer_citation_tags(answer: str, labels: dict[str, str]) -> str:
    if not answer:
        return answer
    out = answer
    sorted_labels = sorted(labels.items(), key=lambda item: len(item[0]), reverse=True)
    for source, label in sorted_labels:
        pattern = re.compile(rf"\[\s*{re.escape(source)}\s*\|\s*-?\d+\s*\]")
        out = pattern.sub(f"[{label}]", out)
    return re.sub(r"\[[^\[\]\n]+\|\s*-?\d+\]", "[Document]", out)


def _sanitize_citations_for_public(citations: list[Citation]) -> list[Citation]:
    labels = _public_source_labels(citations)
    deduped: list[Citation] = []
    seen = set()
    for citation in citations:
        label = labels.get(citation.source, "Document")
        if label in seen:
            continue
        seen.add(label)
        deduped.append(Citation(source=label, chunk_id=-1, score=0.0, text_preview=""))
    return deduped


def _sample_evenly(hits: list[dict], sample_size: int) -> list[dict]:
    total = len(hits)
    if total <= sample_size:
        return hits
    idxs = {round(i * (total - 1) / (sample_size - 1)) for i in range(sample_size)}
    sampled = [hits[i] for i in sorted(idxs)]
    sampled.sort(key=lambda hit: int(hit.get("chunk_id", -1)))
    return sampled


def _dominant_source_hits(hits: list[dict]) -> list[dict]:
    source_counts: dict[str, int] = {}
    for hit in hits:
        source = hit.get("source", "unknown")
        source_counts[source] = source_counts.get(source, 0) + 1
    dominant = max(source_counts.items(), key=lambda pair: pair[1])[0]
    dominant_hits = [hit for hit in hits if hit.get("source", "unknown") == dominant]
    dominant_hits.sort(key=lambda hit: int(hit.get("chunk_id", -1)))
    return dominant_hits


def _diversify_hits_by_source(
    hits: list[dict],
    limit: int,
    per_source_cap: int = 2,
) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    source_order: list[str] = []
    for hit in hits:
        source = str(hit.get("source", "unknown"))
        if source not in buckets:
            buckets[source] = []
            source_order.append(source)
        if len(buckets[source]) < per_source_cap:
            buckets[source].append(hit)

    selected: list[dict] = []
    round_index = 0
    while len(selected) < limit:
        made_progress = False
        for source in source_order:
            items = buckets[source]
            if round_index < len(items):
                selected.append(items[round_index])
                made_progress = True
                if len(selected) >= limit:
                    break
        if not made_progress:
            break
        round_index += 1
    return selected


def _select_contexts(question: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    if not _is_summary_intent(question):
        return hits[: settings.TOP_K]

    max_context = max(settings.TOP_K, 12)
    overlapping = [
        hit for hit in hits if _question_text_overlap(question, hit.get("text", "")) > 0.0
    ]
    if not overlapping:
        overlapping = hits

    if _is_cross_document_intent(question):
        return _diversify_hits_by_source(overlapping, max_context, per_source_cap=3)

    dominant_hits = _dominant_source_hits(overlapping)
    if len(dominant_hits) >= max(4, settings.TOP_K):
        return _sample_evenly(dominant_hits, max_context)
    return overlapping[:max_context]


def _resolve_request_id(request: Request) -> str:
    return request.headers.get("x-request-id") or uuid.uuid4().hex


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _question_metadata(question: str) -> dict[str, str | int]:
    clean = question.strip()
    return {"q_hash": _short_hash(clean), "q_len": len(clean)}


def _error_detail_metadata(detail: object) -> dict[str, str | int]:
    raw = str(detail)
    return {"detail_hash": _short_hash(raw), "detail_len": len(raw)}


def _citation_log_records(contexts: list[dict]) -> list[dict]:
    records: list[dict] = []
    for hit in contexts:
        source = str(hit.get("source", "unknown"))
        records.append(
            {
                "source_hash": _short_hash(source),
                "chunk_id": int(hit.get("chunk_id", -1)),
                "score": float(hit.get("score", 0.0)),
            }
        )
    return records


def _first_sentence(text: str, fallback: str = "No summary available.") -> str:
    clean = _clean_text(text)
    if not clean:
        return fallback
    parts = _split_sentences(clean)
    if not parts:
        return clean[:200]
    return parts[0][:200]


def _indexed_document_entries(limit: int = 50) -> list[dict]:
    client = QdrantClient(url=settings.QDRANT_URL)
    by_source: dict[str, dict] = {}
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
            payload = point.payload or {}
            source = str(payload.get("source", "unknown"))
            chunk_id = int(payload.get("chunk_id", -1))
            text = str(payload.get("text", ""))
            current = by_source.get(source)
            if current is None or chunk_id < current["chunk_id"]:
                by_source[source] = {
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": text,
                }
        if offset is None:
            break

    entries = list(by_source.values())
    entries.sort(key=lambda row: row["source"].lower())
    return entries[:limit]


def _build_document_inventory_result(
    request_id: str,
    state: ChatState,
) -> ChatResponse:
    entries = _indexed_document_entries()
    if not entries:
        return ChatResponse(
            answer=_UNKNOWN_SENTENCE,
            citations=[],
            meta={
                "request_id": request_id,
                "model": settings.CHAT_MODEL,
                "embed_model": settings.EMBED_MODEL,
                "prompt_version": settings.PROMPT_VERSION,
                "retrieve_sec": state.retrieve_sec,
                "rerank_sec": state.rerank_sec,
                "generate_sec": state.generate_sec,
                "top_k": settings.TOP_K,
                "cache_hit": state.cache_hit,
                "fallback_used": state.fallback_used,
                "groundedness": 0.0,
                "inventory_mode": True,
                "indexed_doc_count": 0,
            },
        )

    lines = ["Indexed documents and one-line summaries:"]
    contexts: list[dict] = []
    for entry in entries:
        source = entry["source"]
        chunk_id = entry["chunk_id"]
        summary = _first_sentence(entry.get("text", ""))
        lines.append(f"- {Path(source).name}: {summary} [{source} | {chunk_id}]")
        contexts.append(
            {
                "source": source,
                "chunk_id": chunk_id,
                "score": 1.0,
                "text": entry.get("text", ""),
            }
        )

    return ChatResponse(
        answer="\n".join(lines),
        citations=_build_citations(contexts),
        meta={
            "request_id": request_id,
            "model": settings.CHAT_MODEL,
            "embed_model": settings.EMBED_MODEL,
            "prompt_version": settings.PROMPT_VERSION,
            "retrieve_sec": state.retrieve_sec,
            "rerank_sec": state.rerank_sec,
            "generate_sec": state.generate_sec,
            "top_k": settings.TOP_K,
            "cache_hit": state.cache_hit,
            "fallback_used": state.fallback_used,
            "groundedness": 1.0,
            "inventory_mode": True,
            "indexed_doc_count": len(entries),
        },
    )


def _refresh_runtime_gauges() -> None:
    try:
        docs_count = sum(1 for p in Path("data/docs").rglob("*") if p.is_file())
        set_docs_indexed_total(docs_count)
    except OSError:
        set_docs_indexed_total(0)

    try:
        client = QdrantClient(url=settings.QDRANT_URL)
        result = client.count(collection_name=settings.QDRANT_COLLECTION, exact=False)
        set_qdrant_collection_points(int(result.count))
    except (requests.exceptions.RequestException, UnexpectedResponse, TypeError, ValueError):
        set_qdrant_collection_points(0)


def _cache_response_for_request(
    question: str, request_id: str, state: ChatState
) -> ChatResponse | None:
    if not settings.ENABLE_CACHE:
        return None
    cached = get_cached(question)
    if not cached:
        return None
    state.cache_hit = True
    inc_cache_hits()
    result = ChatResponse(**cached)
    result.meta = {
        **result.meta,
        "request_id": request_id,
        "cache_hit": True,
        "fallback_used": False,
    }
    return result


def _retrieve_hits(question: str, state: ChatState) -> list[dict]:
    retrieve_top_k = max(settings.TOP_K, 18) if _is_summary_intent(question) else settings.TOP_K
    try:
        with timer() as elapsed:
            hits = retrieve(question, top_k=retrieve_top_k)
        state.retrieve_sec = elapsed()
        observe_retrieve_latency(state.retrieve_sec)
        return hits
    except UnexpectedResponse as exc:
        status = getattr(exc, "status_code", None)
        inc_chat_error("qdrant_error")
        if status == 404:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Qdrant collection not found. Run ingestion first: "
                    "`..\\.venv\\Scripts\\python -m app.rag.ingest`"
                ),
            ) from exc
        raise HTTPException(status_code=503, detail=f"Qdrant error: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        inc_chat_error("embedding_unavailable")
        raise HTTPException(
            status_code=503, detail=f"Embedding service unavailable: {exc}"
        ) from exc


def _maybe_rerank(question: str, hits: list[dict], state: ChatState) -> list[dict]:
    if not settings.ENABLE_RERANK or len(hits) <= 1:
        return hits
    rerank_keep = max(settings.TOP_K, 18) if _is_summary_intent(question) else settings.TOP_K
    try:
        with timer() as elapsed:
            ranked_hits = llm_rerank(question, hits, keep=rerank_keep)
        state.rerank_sec = elapsed()
        observe_rerank_latency(state.rerank_sec)
        return ranked_hits
    except (requests.exceptions.RequestException, ValueError):
        inc_rerank_failures()
        return hits


def _generate_answer(question: str, contexts: list[dict], state: ChatState) -> str:
    try:
        with timer() as elapsed:
            answer = generate_answer(question, contexts)
        state.generate_sec = elapsed()
        observe_generate_latency(state.generate_sec)
        return answer
    except requests.exceptions.RequestException as exc:
        inc_chat_error("generation_unavailable")
        raise HTTPException(
            status_code=503, detail=f"Generation service unavailable: {exc}"
        ) from exc


def _apply_fallback_policy(
    question: str, contexts: list[dict], answer: str, state: ChatState
) -> str:
    if _contains_unknown_marker(answer):
        stripped = _strip_unknown_marker(answer)
        if len(stripped.split()) >= 12:
            return stripped
        fallback = _extractive_fallback(question, contexts)
        if fallback:
            state.fallback_used = True
            inc_fallback_answers()
            return fallback
        return _UNKNOWN_SENTENCE

    grounded = _answer_groundedness(answer, contexts)
    if grounded >= settings.FAITHFULNESS_MIN_GROUNDED:
        return answer

    fallback = _extractive_fallback(question, contexts)
    if fallback:
        state.fallback_used = True
        inc_fallback_answers()
        return fallback
    return _UNKNOWN_SENTENCE


def _build_citations(contexts: list[dict]) -> list[Citation]:
    return [
        Citation(
            source=hit["source"],
            chunk_id=hit["chunk_id"],
            score=hit["score"],
            text_preview=hit["text"][:160].replace("\n", " "),
        )
        for hit in contexts
    ]


def _build_result(
    request_id: str,
    answer: str,
    contexts: list[dict],
    state: ChatState,
) -> ChatResponse:
    citations = _build_citations(contexts)
    return ChatResponse(
        answer=answer,
        citations=citations,
        meta={
            "request_id": request_id,
            "model": settings.CHAT_MODEL,
            "embed_model": settings.EMBED_MODEL,
            "prompt_version": settings.PROMPT_VERSION,
            "retrieve_sec": state.retrieve_sec,
            "rerank_sec": state.rerank_sec,
            "generate_sec": state.generate_sec,
            "top_k": settings.TOP_K,
            "cache_hit": state.cache_hit,
            "fallback_used": state.fallback_used,
            "groundedness": _answer_groundedness(answer, contexts),
        },
    )


def _build_guardrail_block_result(
    request_id: str,
    state: ChatState,
    categories: list[str],
    question: str,
) -> ChatResponse:
    for category in categories:
        inc_prompt_attack_blocked(category)
    answer = _guardrail_refusal_answer(question, categories)
    return ChatResponse(
        answer=answer,
        citations=[],
        meta={
            "request_id": request_id,
            "model": settings.CHAT_MODEL,
            "embed_model": settings.EMBED_MODEL,
            "prompt_version": settings.PROMPT_VERSION,
            "retrieve_sec": state.retrieve_sec,
            "rerank_sec": state.rerank_sec,
            "generate_sec": state.generate_sec,
            "top_k": settings.TOP_K,
            "cache_hit": state.cache_hit,
            "fallback_used": state.fallback_used,
            "groundedness": 0.0,
            "guardrail_blocked": True,
            "guardrail_categories": categories,
        },
    )


def _build_out_of_scope_result(
    request_id: str,
    state: ChatState,
    coverage: float,
) -> ChatResponse:
    inc_out_of_scope_block()
    return ChatResponse(
        answer=_UNKNOWN_SENTENCE,
        citations=[],
        meta={
            "request_id": request_id,
            "model": settings.CHAT_MODEL,
            "embed_model": settings.EMBED_MODEL,
            "prompt_version": settings.PROMPT_VERSION,
            "retrieve_sec": state.retrieve_sec,
            "rerank_sec": state.rerank_sec,
            "generate_sec": state.generate_sec,
            "top_k": settings.TOP_K,
            "cache_hit": state.cache_hit,
            "fallback_used": state.fallback_used,
            "groundedness": 0.0,
            "out_of_scope_blocked": True,
            "question_context_coverage": coverage,
        },
    )


def _guardrail_refusal_answer(question: str, categories: list[str]) -> str:
    low = question.lower()
    if "hot-wire" in low or "hot wire" in low:
        return (
            "I can't help with instructions for hot-wiring a car, since that can be "
            "used to bypass vehicle security.\n\n"
            "If you're locked out or having trouble starting your car, try legal and "
            "safe options:\n"
            "- Call roadside assistance (for example AAA or your insurance provider).\n"
            "- Contact a professional automotive locksmith.\n"
            "- Use your manufacturer's roadside service.\n"
            "- Visit a dealership for key fob, immobilizer, or ignition issues.\n\n"
            "If useful, I can explain why modern cars are hard to hot-wire and what "
            "to check when a key or ignition system fails."
        )

    category_messages = [
        (
            {"phishing_or_social_engineering"},
            (
                "I can't help with phishing, impersonation, or credential theft.\n\n"
                "If your goal is defense, I can help with:\n"
                "- Phishing detection checklists.\n"
                "- Secure user-awareness messaging.\n"
                "- Incident response and reporting steps."
            ),
        ),
        (
            {"harmful_content"},
            (
                "I can't help with harmful or illegal instructions.\n\n"
                "If your goal is safety, legal compliance, or defense, I can help with:\n"
                "- Risk awareness and prevention guidance.\n"
                "- Detection and response best practices.\n"
                "- Safe, lawful alternatives."
            ),
        ),
        (
            {"information_exfiltration", "blocked_term"},
            (
                "I can't help with requests to extract secrets, credentials, or sensitive "
                "data.\n\n"
                "I can help with defensive alternatives:\n"
                "- Secret scanning and rotation practices.\n"
                "- Access-control hardening.\n"
                "- Audit logging and anomaly detection."
            ),
        ),
        (
            {"tool_or_system_abuse", "jailbreak"},
            (
                "I can't help with bypassing safeguards, abusing systems, or harmful "
                "operational instructions.\n\n"
                "I can help with secure and legal alternatives:\n"
                "- Hardening and least-privilege guidance.\n"
                "- Safe testing plans in authorized environments.\n"
                "- Defensive monitoring and recovery playbooks."
            ),
        ),
    ]
    seen_categories = set(categories)
    for category_set, message in category_messages:
        if seen_categories & category_set:
            return message

    return (
        "I can't help with that request.\n\n"
        "If your goal is defensive or educational, I can provide safe guidance and "
        "best practices."
    )


def _apply_public_sanitization(result: ChatResponse) -> ChatResponse:
    if not settings.PUBLIC_RESPONSE_SANITIZE:
        return result
    labels = _public_source_labels(result.citations)
    result.answer = _sanitize_answer_citation_tags(result.answer, labels)
    result.citations = _sanitize_citations_for_public(result.citations)
    result.meta = {**result.meta, "public_response_sanitized": True}
    return result


def _log_chat_start(request_id: str, question: str) -> None:
    log_event(
        {
            "type": "chat_start",
            "request_id": request_id,
            "model": settings.CHAT_MODEL,
            "embed_model": settings.EMBED_MODEL,
            **_question_metadata(question),
        }
    )


def _log_guardrail_block(
    request_id: str, question: str, categories: list[str]
) -> None:
    log_event(
        {
            "type": "guardrail_block",
            "request_id": request_id,
            "categories": categories,
            **_question_metadata(question),
        }
    )


def _log_out_of_scope_block(request_id: str, question: str, coverage: float) -> None:
    log_event(
        {
            "type": "out_of_scope_block",
            "request_id": request_id,
            "question_context_coverage": coverage,
            **_question_metadata(question),
        }
    )


def _log_chat_success(
    request_id: str, question: str, result: ChatResponse, contexts: list[dict]
) -> None:
    log_event(
        {
            "type": "chat_success",
            "request_id": request_id,
            "meta": result.meta,
            "citations": _citation_log_records(contexts),
            **_question_metadata(question),
        }
    )


def _log_chat_end(request_id: str, question: str, state: ChatState) -> None:
    total = time.time() - state.started
    observe_chat_latency(total)
    log_event(
        {
            "type": "chat_end",
            "request_id": request_id,
            "total_sec": total,
            "retrieve_sec": state.retrieve_sec,
            "rerank_sec": state.rerank_sec,
            "generate_sec": state.generate_sec,
            "cache_hit": state.cache_hit,
            "fallback_used": state.fallback_used,
            **_question_metadata(question),
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics(_: None = Depends(require_api_key)) -> Response:
    return Response(content=render_metrics(), media_type=metrics_content_type())


def _validate_question_or_raise(question: str) -> None:
    if not question:
        inc_chat_error("bad_request")
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not _is_meaningful_question(question):
        inc_chat_error("bad_request")
        raise HTTPException(
            status_code=400,
            detail="Please enter a meaningful question.",
        )


def _run_chat_pipeline(question: str, request_id: str, state: ChatState) -> ChatResponse:
    inventory = _maybe_inventory_result(question, request_id, state)
    if inventory is not None:
        return inventory

    guardrail_result = _maybe_guardrail_block(question, request_id, state)
    if guardrail_result is not None:
        return guardrail_result

    cached = _cache_response_for_request(question, request_id, state)
    if cached:
        return cached

    hits = _retrieve_hits(question, state)
    hits = _maybe_rerank(question, hits, state)
    contexts = _select_contexts(question, hits)
    scope_result = _maybe_out_of_scope_result(question, request_id, state, contexts)
    if scope_result is not None:
        return scope_result

    answer = _generate_answer(question, contexts, state)
    answer = _apply_fallback_policy(question, contexts, answer, state)
    if answer != _UNKNOWN_SENTENCE:
        answer = _ensure_inline_citation(answer, contexts)

    result = _build_result(request_id, answer, contexts, state)
    result = _apply_public_sanitization(result)
    _log_chat_success(request_id, question, result, contexts)

    if settings.ENABLE_CACHE:
        set_cached(question, result.model_dump())
    return result


def _maybe_inventory_result(
    question: str,
    request_id: str,
    state: ChatState,
) -> ChatResponse | None:
    if not _is_document_inventory_intent(question):
        return None
    try:
        result = _build_document_inventory_result(
            request_id=request_id,
            state=state,
        )
        result = _apply_public_sanitization(result)
        _log_chat_success(request_id, question, result, contexts=[])
        return result
    except UnexpectedResponse as exc:
        inc_chat_error("qdrant_error")
        raise HTTPException(status_code=503, detail=f"Qdrant error: {exc}") from exc
    except requests.exceptions.RequestException as exc:
        inc_chat_error("qdrant_unavailable")
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant service unavailable: {exc}",
        ) from exc
    except Exception as exc:
        # qdrant-client may wrap connection failures in custom exceptions.
        if "ResponseHandlingException" in type(exc).__name__:
            inc_chat_error("qdrant_unavailable")
            raise HTTPException(
                status_code=503,
                detail=f"Qdrant service unavailable: {exc}",
            ) from exc
        raise


def _maybe_guardrail_block(
    question: str,
    request_id: str,
    state: ChatState,
) -> ChatResponse | None:
    if not settings.ENABLE_PROMPT_GUARDRAILS:
        return None

    risk = analyze_question_risk(question)
    if not risk["blocked"]:
        return None

    result = _build_guardrail_block_result(
        request_id=request_id,
        state=state,
        categories=risk["categories"],
        question=question,
    )
    _log_guardrail_block(request_id, question, risk["categories"])
    _log_chat_success(request_id, question, result, contexts=[])
    return result


def _maybe_out_of_scope_result(
    question: str,
    request_id: str,
    state: ChatState,
    contexts: list[dict],
) -> ChatResponse | None:
    if not settings.ENABLE_OUT_OF_SCOPE_GUARD:
        return None

    coverage = _question_context_coverage(question, contexts)
    if coverage >= settings.OUT_OF_SCOPE_MIN_COVERAGE:
        return None

    result = _build_out_of_scope_result(
        request_id=request_id,
        state=state,
        coverage=coverage,
    )
    _log_out_of_scope_block(request_id, question, coverage)
    _log_chat_success(request_id, question, result, contexts=[])
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, _: None = Depends(require_api_key)) -> ChatResponse:
    request_id = _resolve_request_id(request)
    state = ChatState(started=time.time())
    question = req.question.strip()
    _refresh_runtime_gauges()

    inc_chat_requests()
    _validate_question_or_raise(question)

    _log_chat_start(request_id, question)

    try:
        return _run_chat_pipeline(question, request_id, state)
    except HTTPException as exc:
        inc_chat_error(f"http_{exc.status_code}")
        log_event(
            {
                "type": "chat_error",
                "request_id": request_id,
                "status_code": exc.status_code,
                **_question_metadata(question),
                **_error_detail_metadata(exc.detail),
            }
        )
        raise
    except Exception as exc:
        inc_chat_error(type(exc).__name__)
        log_event(
            {
                "type": "chat_error",
                "request_id": request_id,
                "status_code": 500,
                "error_type": type(exc).__name__,
                **_question_metadata(question),
                **_error_detail_metadata(exc),
            }
        )
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        _log_chat_end(request_id, question, state)


def main() -> None:
    uvicorn.run(
        "app.__main__:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()
