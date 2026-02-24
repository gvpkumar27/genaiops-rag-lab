import requests
from app.config import settings
from app.rag.prompts import SYSTEM_PROMPT, build_user_prompt

def generate_answer(question: str, contexts: list[dict]) -> str:
    user_prompt = build_user_prompt(question, contexts)
    chat_url = f"{settings.OLLAMA_BASE_URL}/api/chat"

    r = requests.post(chat_url, json={
        "model": settings.CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": settings.OLLAMA_TEMPERATURE,
            "top_p": settings.OLLAMA_TOP_P,
            "seed": settings.OLLAMA_SEED,
        },
    }, timeout=240)
    if r.status_code == 404:
        # Backward compatibility for older Ollama servers.
        gen_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        fallback = requests.post(
            gen_url,
            json={
                "model": settings.CHAT_MODEL,
                "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                "stream": False,
                "options": {
                    "temperature": settings.OLLAMA_TEMPERATURE,
                    "top_p": settings.OLLAMA_TOP_P,
                    "seed": settings.OLLAMA_SEED,
                },
            },
            timeout=240,
        )
        fallback.raise_for_status()
        return fallback.json()["response"]

    r.raise_for_status()
    return r.json()["message"]["content"]
