import re
import time
import uuid

import requests
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.ops.logging import log_event
from app.ops.metrics import (
    inc_cache_hits,
    inc_chat_error,
    inc_chat_requests,
    inc_fallback_answers,
    inc_rerank_failures,
    metrics_content_type,
    observe_chat_latency,
    observe_generate_latency,
    observe_rerank_latency,
    observe_retrieve_latency,
    render_metrics,
    timer,
)
from app.rag.cache import get_cached, set_cached
from app.rag.generate import generate_answer
from app.rag.rerank import llm_rerank
from app.rag.retrieve import retrieve
from app.schemas import ChatRequest, ChatResponse, Citation

app = FastAPI(title="LocalDocChat")

def _is_local_bind_host(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1"}


@app.on_event("startup")
def startup_security_check() -> None:
    if (
        settings.REQUIRE_API_KEY_ON_NON_LOCALHOST
        and not settings.API_KEY
        and not _is_local_bind_host(settings.API_HOST)
    ):
        raise RuntimeError(
            "Unsafe configuration: API_HOST is non-localhost but API_KEY is empty. "
            "Set API_KEY or set REQUIRE_API_KEY_ON_NON_LOCALHOST=false only for trusted local testing."
        )


def require_api_key(request: Request) -> None:
    if (
        settings.REQUIRE_API_KEY_ON_NON_LOCALHOST
        and not settings.API_KEY
        and not _is_local_bind_host(settings.API_HOST)
    ):
        raise HTTPException(status_code=503, detail="Server misconfiguration: API key required for non-localhost bind")
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


def _tok(text: str) -> set[str]:
    toks = set(re.findall(r"[a-z0-9]{3,}", text.lower()))
    return {t for t in toks if t not in _STOPWORDS}


def _question_text_overlap(question: str, text: str) -> float:
    q = _tok(question)
    if not q:
        return 0.0
    t = _tok(text)
    return len(q & t) / max(1, len(q))


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_summary_intent(question: str) -> bool:
    return bool(re.search(r"\b(summarize|summarise|summary|overview|highlights|features)\b", question.lower()))


def _is_good_fallback_sentence(sent: str) -> bool:
    words = sent.split()
    if len(words) < 6 or len(words) > 36:
        return False

    low = sent.lower()
    noise_markers = (
        "article tags",
        "last updated",
        "sign in",
        "output",
        "driver code",
        "formatted current date and time",
    )
    if any(m in low for m in noise_markers):
        return False

    alpha_count = sum(ch.isalpha() for ch in sent)
    ratio = alpha_count / max(1, len(sent))
    return ratio >= 0.65


def _extractive_fallback(question: str, hits: list[dict]) -> str | None:
    q = _tok(question)
    if not q:
        return None

    ranked: list[tuple[int, int, str, str, int]] = []
    for h in hits:
        text = _clean_text(h.get("text", ""))
        source = h.get("source", "unknown")
        chunk_id = int(h.get("chunk_id", -1))
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
        for p in parts:
            p_toks = _tok(p)
            score = len(q & p_toks)
            words = len(p.split())
            keyword_score = score / max(1, len(q))
            if score >= 2 and keyword_score >= 0.3 and words >= 6 and _is_good_fallback_sentence(p):
                ranked.append((score, words, p, source, chunk_id))

    if not ranked:
        summary_intent = _is_summary_intent(question)
        if not summary_intent:
            return None

        # Summary queries often have broad wording with weak token overlap.
        picked: list[tuple[str, str, int]] = []
        seen = set()
        for h in hits[:5]:
            text = _clean_text(h.get("text", ""))
            source = h.get("source", "unknown")
            chunk_id = int(h.get("chunk_id", -1))
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
            for p in parts:
                key = (source, chunk_id, p[:80])
                if key in seen or not _is_good_fallback_sentence(p):
                    continue
                seen.add(key)
                picked.append((p, source, chunk_id))
                if len(picked) == 5:
                    break
            if len(picked) == 5:
                break

        if not picked:
            return None
        lines = [f"- {sent} [{source} | {chunk_id}]" for sent, source, chunk_id in picked]
        return "Summary from the uploaded documents:\n" + "\n".join(lines)

    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    picked = []
    seen = set()
    for score, words, sent, source, chunk_id in ranked:
        key = (source, chunk_id, sent[:80])
        if key in seen:
            continue
        seen.add(key)
        picked.append((sent, source, chunk_id))
        if len(picked) == 3:
            break

    if not picked:
        return None

    lines = [f"- {sent} [{source} | {chunk_id}]" for sent, source, chunk_id in picked]
    return "Based on the uploaded documents:\n" + "\n".join(lines)


def _answer_groundedness(answer: str, hits: list[dict]) -> float:
    a_toks = _tok(answer)
    if not a_toks:
        return 0.0
    ctx_toks = set()
    for h in hits:
        ctx_toks |= _tok(h.get("text", ""))
    if not ctx_toks:
        return 0.0
    return len(a_toks & ctx_toks) / max(1, len(a_toks))


def _has_inline_citation(answer: str) -> bool:
    return bool(re.search(r"\[[^\[\]\n]+\|\s*-?\d+\]", answer))


def _top_citation_tag(hits: list[dict]) -> str:
    if not hits:
        return "[unknown | -1]"
    h = hits[0]
    return f"[{h.get('source', 'unknown')} | {int(h.get('chunk_id', -1))}]"


def _ensure_inline_citation(answer: str, hits: list[dict]) -> str:
    if _has_inline_citation(answer):
        return answer
    tag = _top_citation_tag(hits)
    out = answer.strip()
    if not out:
        return out
    return f"{out} {tag}"


def _contains_unknown_marker(answer: str) -> bool:
    return _UNKNOWN_SENTENCE.lower() in answer.strip().lower()


def _strip_unknown_marker(answer: str) -> str:
    out = re.sub(re.escape(_UNKNOWN_SENTENCE), "", answer, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip(" \n\r\t-")


def _select_contexts(question: str, hits: list[dict]) -> list[dict]:
    if not hits:
        return []
    if not _is_summary_intent(question):
        return hits[: settings.TOP_K]

    max_ctx = max(settings.TOP_K, 12)
    with_overlap = [h for h in hits if _question_text_overlap(question, h.get("text", "")) > 0.0]
    if not with_overlap:
        with_overlap = hits

    # For summary queries, prioritize the dominant relevant source to avoid cross-document drift.
    src_counts: dict[str, int] = {}
    for h in with_overlap:
        src = h.get("source", "unknown")
        src_counts[src] = src_counts.get(src, 0) + 1
    dominant = max(src_counts.items(), key=lambda kv: kv[1])[0]
    dominant_hits = [h for h in with_overlap if h.get("source", "unknown") == dominant]
    dominant_hits.sort(key=lambda h: int(h.get("chunk_id", -1)))
    if len(dominant_hits) >= max(4, settings.TOP_K):
        if len(dominant_hits) <= max_ctx:
            return dominant_hits
        # Spread samples across the document to improve feature coverage.
        n = len(dominant_hits)
        idxs = {round(i * (n - 1) / (max_ctx - 1)) for i in range(max_ctx)}
        sampled = [dominant_hits[i] for i in sorted(idxs)]
        sampled.sort(key=lambda h: int(h.get("chunk_id", -1)))
        return sampled
    return with_overlap[:max_ctx]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics(_: None = Depends(require_api_key)):
    return Response(content=render_metrics(), media_type=metrics_content_type())


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, _: None = Depends(require_api_key)):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    started = time.time()
    retrieve_sec = 0.0
    rerank_sec = 0.0
    generate_sec = 0.0
    cache_hit = False
    fallback_used = False

    inc_chat_requests()
    question = req.question.strip()
    if not question:
        inc_chat_error("bad_request")
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    log_event(
        {
            "type": "chat_start",
            "request_id": request_id,
            "q": question,
            "model": settings.CHAT_MODEL,
            "embed_model": settings.EMBED_MODEL,
        }
    )

    try:
        if settings.ENABLE_CACHE:
            cached = get_cached(question)
            if cached:
                cache_hit = True
                inc_cache_hits()
                result = ChatResponse(**cached)
                result.meta = {
                    **result.meta,
                    "request_id": request_id,
                    "cache_hit": True,
                    "fallback_used": False,
                }
                return result

        try:
            retrieve_top_k = max(settings.TOP_K, 18) if _is_summary_intent(question) else settings.TOP_K
            with timer() as t_retrieve:
                hits = retrieve(question, top_k=retrieve_top_k)
            retrieve_sec = t_retrieve()
            observe_retrieve_latency(retrieve_sec)
        except UnexpectedResponse as e:
            status = getattr(e, "status_code", None)
            inc_chat_error("qdrant_error")
            if status == 404:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Qdrant collection not found. Run ingestion first: "
                        "`..\\.venv\\Scripts\\python -m app.rag.ingest`"
                    ),
                ) from e
            raise HTTPException(status_code=503, detail=f"Qdrant error: {e}") from e
        except requests.exceptions.RequestException as e:
            inc_chat_error("embedding_unavailable")
            raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {e}") from e

        if settings.ENABLE_RERANK and len(hits) > 1:
            try:
                with timer() as t_rerank:
                    rerank_keep = max(settings.TOP_K, 18) if _is_summary_intent(question) else settings.TOP_K
                    hits = llm_rerank(question, hits, keep=rerank_keep)
                rerank_sec = t_rerank()
                observe_rerank_latency(rerank_sec)
            except (requests.exceptions.RequestException, ValueError):
                inc_rerank_failures()

        contexts = _select_contexts(question, hits)

        try:
            with timer() as t_gen:
                answer = generate_answer(question, contexts)
            generate_sec = t_gen()
            observe_generate_latency(generate_sec)
        except requests.exceptions.RequestException as e:
            inc_chat_error("generation_unavailable")
            raise HTTPException(status_code=503, detail=f"Generation service unavailable: {e}") from e

        if _contains_unknown_marker(answer):
            stripped = _strip_unknown_marker(answer)
            if len(stripped.split()) >= 12:
                answer = stripped
            else:
                fallback = _extractive_fallback(question, contexts)
                if fallback:
                    fallback_used = True
                    inc_fallback_answers()
                    answer = fallback
                else:
                    answer = _UNKNOWN_SENTENCE
        else:
            grounded = _answer_groundedness(answer, contexts)
            if grounded < settings.FAITHFULNESS_MIN_GROUNDED:
                fallback = _extractive_fallback(question, contexts)
                if fallback:
                    fallback_used = True
                    inc_fallback_answers()
                    answer = fallback
                else:
                    answer = _UNKNOWN_SENTENCE

        if answer != _UNKNOWN_SENTENCE:
            answer = _ensure_inline_citation(answer, contexts)

        citations = [
            Citation(
                source=h["source"],
                chunk_id=h["chunk_id"],
                score=h["score"],
                text_preview=h["text"][:160].replace("\n", " "),
            )
            for h in contexts
        ]

        result = ChatResponse(
            answer=answer,
            citations=citations,
            meta={
                "request_id": request_id,
                "model": settings.CHAT_MODEL,
                "embed_model": settings.EMBED_MODEL,
                "prompt_version": settings.PROMPT_VERSION,
                "retrieve_sec": retrieve_sec,
                "rerank_sec": rerank_sec,
                "generate_sec": generate_sec,
                "top_k": settings.TOP_K,
                "cache_hit": cache_hit,
                "fallback_used": fallback_used,
                "groundedness": _answer_groundedness(answer, contexts),
            },
        )

        log_event(
            {
                "type": "chat_success",
                "request_id": request_id,
                "q": question,
                "meta": result.meta,
                "citations": [c.model_dump() for c in citations],
            }
        )

        if settings.ENABLE_CACHE:
            set_cached(question, result.model_dump())

        return result
    except HTTPException as e:
        inc_chat_error(f"http_{e.status_code}")
        log_event(
            {
                "type": "chat_error",
                "request_id": request_id,
                "q": question,
                "status_code": e.status_code,
                "detail": str(e.detail),
            }
        )
        raise
    except Exception as e:
        inc_chat_error(type(e).__name__)
        log_event(
            {
                "type": "chat_error",
                "request_id": request_id,
                "q": question,
                "status_code": 500,
                "error_type": type(e).__name__,
                "detail": str(e),
            }
        )
        raise HTTPException(status_code=500, detail="Internal server error") from e
    finally:
        total = time.time() - started
        observe_chat_latency(total)
        log_event(
            {
                "type": "chat_end",
                "request_id": request_id,
                "q": question,
                "total_sec": total,
                "retrieve_sec": retrieve_sec,
                "rerank_sec": rerank_sec,
                "generate_sec": generate_sec,
                "cache_hit": cache_hit,
                "fallback_used": fallback_used,
            }
        )


def main() -> None:
    import uvicorn

    uvicorn.run("app.__main__:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)


if __name__ == "__main__":
    main()
