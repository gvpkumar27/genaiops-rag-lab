import re

from app.config import settings


_FILLERS = {
    "a",
    "an",
    "the",
    "please",
    "tell",
    "me",
    "about",
    "explain",
    "what",
    "is",
    "are",
    "could",
    "would",
    "can",
    "you",
    "in",
}


def _normalize_token(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("s"):
        return tok[:-1]
    return tok


def normalized_question(question: str) -> str:
    base = question.strip().lower()
    if not settings.STABLE_QUERY_NORMALIZATION:
        return re.sub(r"\s+", " ", base)

    base = re.sub(r"[^a-z0-9\s]", " ", base)
    toks = [_normalize_token(t) for t in re.findall(r"[a-z0-9]+", base) if t not in _FILLERS]
    if not toks:
        return ""
    return " ".join(sorted(toks))
