import re
from qdrant_client import QdrantClient

from app.config import settings
from app.rag.embeddings import embed_text


def _merge_fragmented_letters(text: str) -> str:
    pattern = re.compile(r"\b(?:[A-Za-z]{1,2}\s+){3,}[A-Za-z]{1,12}\b")

    def repl(match: re.Match[str]) -> str:
        tokens = match.group(0).split()
        single_count = sum(1 for t in tokens if len(t) == 1)
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


def retrieve(question: str, top_k: int | None = None):
    client = QdrantClient(url=settings.QDRANT_URL)
    qvec = embed_text(question)

    final_k = top_k or settings.TOP_K
    candidate_k = max(final_k, settings.RETRIEVE_CANDIDATES, final_k * 4)

    semantic_res = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=qvec,
        limit=candidate_k,
    )

    q_tokens = _tokenize(question)
    by_key = {}
    for r in semantic_res:
        p = r.payload or {}
        text = p.get("text", "")
        t_tokens = _tokenize(text)
        overlap = len(q_tokens & t_tokens)
        keyword_score = overlap / max(1, len(q_tokens))
        semantic_score = float(r.score)
        blended_score = semantic_score + (0.25 * keyword_score)
        key = (p.get("source", "unknown"), int(p.get("chunk_id", -1)))
        by_key[key] = {
            "score": semantic_score,
            "rank_score": blended_score,
            "source": key[0],
            "chunk_id": key[1],
            "text": text,
        }

    # Lexical fallback over full collection improves recall for exact document terms.
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
        for pt in points:
            p = pt.payload or {}
            text = p.get("text", "")
            t_tokens = _tokenize(text)
            overlap = len(q_tokens & t_tokens)
            if overlap == 0:
                continue
            keyword_score = overlap / max(1, len(q_tokens))
            # If chunk also came from semantic search, keep stronger score.
            key = (p.get("source", "unknown"), int(p.get("chunk_id", -1)))
            current = by_key.get(key)
            lexical_rank = 0.15 + (0.55 * keyword_score)
            if current is None or lexical_rank > current["rank_score"]:
                by_key[key] = {
                    "score": current["score"] if current else 0.0,
                    "rank_score": max(lexical_rank, current["rank_score"]) if current else lexical_rank,
                    "source": key[0],
                    "chunk_id": key[1],
                    "text": text,
                }
        if offset is None:
            break

    hits = list(by_key.values())
    hits.sort(key=lambda h: h["rank_score"], reverse=True)
    return hits[:final_k]
