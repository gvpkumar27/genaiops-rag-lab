import json, requests
from app.config import settings

def llm_rerank(question: str, hits: list[dict], keep: int = 5) -> list[dict]:
    chat_url = f"{settings.OLLAMA_BASE_URL}/api/chat"
    passages = "\n\n".join([f"[{i}] {h['text'][:700]}" for i, h in enumerate(hits)])

    prompt = f"""
Pick the most useful passages to answer.

Question: {question}

Passages:
{passages}

Return ONLY JSON array of indices best-first. Example: [2,0,1]
""".strip()

    r = requests.post(chat_url, json={
        "model": settings.CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": settings.OLLAMA_TEMPERATURE,
            "top_p": settings.OLLAMA_TOP_P,
            "seed": settings.OLLAMA_SEED,
        },
    }, timeout=180)
    if r.status_code == 404:
        gen_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        fallback = requests.post(
            gen_url,
            json={
                "model": settings.CHAT_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": settings.OLLAMA_TEMPERATURE,
                    "top_p": settings.OLLAMA_TOP_P,
                    "seed": settings.OLLAMA_SEED,
                },
            },
            timeout=180,
        )
        fallback.raise_for_status()
        content = fallback.json()["response"].strip()
    else:
        r.raise_for_status()
        content = r.json()["message"]["content"].strip()

    idxs = json.loads(content)
    idxs = [i for i in idxs if isinstance(i, int) and 0 <= i < len(hits)]
    out = [hits[i] for i in idxs][:keep]
    return out if out else hits[:keep]
