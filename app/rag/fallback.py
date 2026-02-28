"""Compatibility helpers for extractive fallback behavior."""

from app.__main__ import _extractive_fallback, _is_good_fallback_sentence


def is_good_fallback_sentence(sentence: str) -> bool:
    return _is_good_fallback_sentence(sentence)


def extractive_fallback(question: str, hits: list[dict]) -> str | None:
    return _extractive_fallback(question, hits)
