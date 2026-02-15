import re
import time
import uuid

import requests
from fastapi import FastAPI, HTTPException, Request, Response
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


def _tok(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
            score = len(q & _tok(p))
            words = len(p.split())
            if score > 0 and words >= 6:
                ranked.append((score, words, p, source, chunk_id))

    if not ranked:
        return None

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=render_metrics(), media_type=metrics_content_type())


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
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
            with timer() as t_retrieve:
                hits = retrieve(question)
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
                    hits = llm_rerank(question, hits, keep=settings.TOP_K)
                rerank_sec = t_rerank()
                observe_rerank_latency(rerank_sec)
            except (requests.exceptions.RequestException, ValueError):
                inc_rerank_failures()

        try:
            with timer() as t_gen:
                answer = generate_answer(question, hits)
            generate_sec = t_gen()
            observe_generate_latency(generate_sec)
        except requests.exceptions.RequestException as e:
            inc_chat_error("generation_unavailable")
            raise HTTPException(status_code=503, detail=f"Generation service unavailable: {e}") from e

        ans_low = answer.strip().lower()
        if ans_low.startswith("i don't know based on the provided documents") or len(answer.split()) < 12:
            fallback = _extractive_fallback(question, hits)
            if fallback:
                fallback_used = True
                inc_fallback_answers()
                answer = fallback

        citations = [
            Citation(
                source=h["source"],
                chunk_id=h["chunk_id"],
                score=h["score"],
                text_preview=h["text"][:160].replace("\n", " "),
            )
            for h in hits
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

    uvicorn.run("app.__main__:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
