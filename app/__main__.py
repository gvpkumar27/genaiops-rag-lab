import requests
import re
from fastapi import FastAPI, HTTPException
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import settings
from app.ops.logging import log_event
from app.ops.metrics import timer
from app.rag.cache import get_cached, set_cached
from app.rag.generate import generate_answer
from app.rag.rerank import llm_rerank
from app.rag.retrieve import retrieve
from app.schemas import ChatRequest, ChatResponse, Citation

app = FastAPI(title="LocalDocChat")


def _tok(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _extractive_fallback(question: str, hits: list[dict]) -> str | None:
    q = _tok(question)
    if not q:
        return None

    ranked: list[tuple[int, str, str, int]] = []
    for h in hits:
        text = h.get("text", "")
        source = h.get("source", "unknown")
        chunk_id = int(h.get("chunk_id", -1))
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
        for p in parts:
            score = len(q & _tok(p))
            if score > 0:
                ranked.append((score, p, source, chunk_id))

    if not ranked:
        return None

    ranked.sort(key=lambda x: x[0], reverse=True)
    picked = []
    seen = set()
    for score, sent, source, chunk_id in ranked:
        key = (source, chunk_id, sent[:80])
        if key in seen:
            continue
        seen.add(key)
        picked.append((sent, source, chunk_id))
        if len(picked) == 2:
            break

    if not picked:
        return None

    lines = [f"{sent} [{source} | {chunk_id}]" for sent, source, chunk_id in picked]
    return " ".join(lines)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        if settings.ENABLE_CACHE:
            cached = get_cached(question)
            if cached:
                return ChatResponse(**cached)

        try:
            with timer() as t_retrieve:
                hits = retrieve(question)
        except UnexpectedResponse as e:
            status = getattr(e, "status_code", None)
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
            raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {e}") from e

        if settings.ENABLE_RERANK and len(hits) > 1:
            try:
                hits = llm_rerank(question, hits, keep=settings.TOP_K)
            except requests.exceptions.RequestException as e:
                # Degrade gracefully if reranker is unavailable/slow.
                # We can still answer using retrieved passages.
                pass
            except ValueError as e:
                # Ignore malformed rerank output and keep original retrieval order.
                pass

        try:
            with timer() as t_gen:
                answer = generate_answer(question, hits)
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Generation service unavailable: {e}") from e

        if answer.strip().lower().startswith("i don't know based on the provided documents"):
            fallback = _extractive_fallback(question, hits)
            if fallback:
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
                "model": settings.CHAT_MODEL,
                "embed_model": settings.EMBED_MODEL,
                "prompt_version": settings.PROMPT_VERSION,
                "retrieve_sec": t_retrieve(),
                "generate_sec": t_gen(),
                "top_k": settings.TOP_K,
            },
        )

        log_event(
            {
                "type": "chat",
                "q": question,
                "meta": result.meta,
                "citations": [c.model_dump() for c in citations],
            }
        )

        if settings.ENABLE_CACHE:
            set_cached(question, result.model_dump())

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unhandled chat error: {type(e).__name__}: {e}") from e


def main() -> None:
    import uvicorn

    uvicorn.run("app.__main__:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
