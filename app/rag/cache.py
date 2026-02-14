import hashlib, json
from pathlib import Path
from app.config import settings

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _key(question: str) -> str:
    raw = json.dumps({"q": question, "m": settings.CHAT_MODEL, "p": settings.PROMPT_VERSION}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def get_cached(question: str):
    fp = CACHE_DIR / f"{_key(question)}.json"
    if fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))
    return None

def set_cached(question: str, value: dict):
    fp = CACHE_DIR / f"{_key(question)}.json"
    fp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
