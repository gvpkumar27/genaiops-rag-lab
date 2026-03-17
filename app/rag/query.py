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

_TYPO_CORRECTIONS = {
    "faver": "fever",
    "fevr": "fever",
    "mahcine": "machine",
    "lernning": "learning",
    "retrival": "retrieval",
    "augumented": "augmented",
    "genration": "generation",
}


def _normalize_token(tok: str) -> str:
    tok = _TYPO_CORRECTIONS.get(tok, tok)
    if len(tok) > 4 and tok.endswith("s"):
        return tok[:-1]
    return tok


def _normalized_tokens(question: str) -> list[str]:
    base = question.strip().lower()
    base = re.sub(r"[^a-z0-9\s]", " ", base)
    return [_normalize_token(t) for t in re.findall(r"[a-z0-9]+", base) if t not in _FILLERS]


def normalized_question(question: str) -> str:
    base = question.strip().lower()
    if not settings.STABLE_QUERY_NORMALIZATION:
        return re.sub(r"\s+", " ", base)

    toks = _normalized_tokens(question)
    if not toks:
        return ""
    return " ".join(sorted(toks))


def rewrite_question_variants(question: str) -> list[str]:
    variants: list[str] = []
    stripped = re.sub(r"\s+", " ", question.strip())
    if stripped:
        variants.append(stripped)

    normalized = normalized_question(question)
    if normalized and normalized not in variants:
        variants.append(normalized)

    if settings.ENABLE_QUERY_REWRITE:
        corrected_tokens = _normalized_tokens(question)
        corrected = " ".join(corrected_tokens).strip()
        if corrected and corrected not in variants:
            variants.append(corrected)

    return variants
