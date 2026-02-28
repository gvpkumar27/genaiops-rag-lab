"""Dataset contamination checks used by evaluation tests."""

import re
from difflib import SequenceMatcher


def _normalize_question(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    return re.sub(r"[^\w\s]", "", text)


def _find_exact_duplicates(splits: dict[str, list[dict]]) -> list[tuple[str, str, str]]:
    seen: dict[str, str] = {}
    issues: list[tuple[str, str, str]] = []
    for split_name, rows in splits.items():
        for row in rows:
            question = _normalize_question(str(row.get("question", "")))
            if not question:
                continue
            existing_split = seen.get(question)
            if existing_split and existing_split != split_name:
                issues.append((existing_split, split_name, question))
                continue
            seen[question] = split_name
    return issues


def _find_near_duplicates(
    splits: dict[str, list[dict]], threshold: float = 0.85
) -> list[tuple[str, str, str, str, float]]:
    named_questions: list[tuple[str, str]] = []
    for split_name, rows in splits.items():
        for row in rows:
            question = _normalize_question(str(row.get("question", "")))
            if question:
                named_questions.append((split_name, question))

    issues: list[tuple[str, str, str, str, float]] = []
    for idx, (split_a, q_a) in enumerate(named_questions):
        for split_b, q_b in named_questions[idx + 1 :]:
            if split_a == split_b:
                continue
            score = SequenceMatcher(None, q_a, q_b).ratio()
            if score >= threshold:
                issues.append((split_a, split_b, q_a, q_b, score))
    return issues
