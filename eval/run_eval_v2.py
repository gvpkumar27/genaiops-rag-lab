"""Reusable scoring helpers for evaluation tests."""

import re
from collections.abc import Iterable


UNKNOWN_SENTENCE = "I don't know based on the provided documents."


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", text.lower()))


def _keyword_coverage(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_low = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_low)
    return hits / len(expected_keywords)


def _lexical_f1(answer: str, reference: str) -> float | None:
    answer_tokens = _tokens(answer)
    reference_tokens = _tokens(reference)
    if not answer_tokens or not reference_tokens:
        return None
    overlap = len(answer_tokens & reference_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(answer_tokens)
    recall = overlap / len(reference_tokens)
    return (2 * precision * recall) / max(1e-12, precision + recall)


def _to_citation_key(value: str | dict) -> str | None:
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, dict):
        source = str(value.get("source", "unknown"))
        chunk_id = int(value.get("chunk_id", -1))
        return f"{source}|{chunk_id}".lower()
    return None


def _citation_metrics(citations: list[dict], gold: Iterable[str | dict]) -> tuple[float, float]:
    pred_keys = {_to_citation_key(c) for c in citations}
    pred_keys.discard(None)
    gold_keys = {_to_citation_key(g) for g in gold}
    gold_keys.discard(None)

    if not gold_keys:
        recall = 1.0
    else:
        recall = len(pred_keys & gold_keys) / len(gold_keys)

    if not pred_keys:
        precision = 0.0
    else:
        precision = len(pred_keys & gold_keys) / len(pred_keys)
    return recall, precision


def _groundedness(answer: str, citations: list[dict]) -> float:
    answer_tokens = _tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for citation in citations:
        context_tokens |= _tokens(str(citation.get("text_preview", "")))
    if not context_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _unknown_behavior(answer: str, should_be_unknown: bool) -> float:
    is_unknown = answer.strip().lower() == UNKNOWN_SENTENCE.lower()
    if should_be_unknown and is_unknown:
        return 1.0
    if should_be_unknown and not is_unknown:
        return 0.0
    if not should_be_unknown and is_unknown:
        return 0.0
    return 1.0
