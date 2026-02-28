from app.config import settings
from app.ops.http import post_json


def embed_text(text: str) -> list[float]:
    url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"
    r = post_json(
        url,
        payload={"model": settings.EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_text(t) for t in texts]
