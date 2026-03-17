from pathlib import Path

from app.config import settings
from app.rag.intent import (
    COMPARISON,
    DEFINITION,
    EXAMPLES,
    SUMMARY,
    WHEN_TO_USE,
    detect_answer_intent,
)


def _load_system_prompt() -> str:
    if settings.SYSTEM_PROMPT_FILE:
        path = Path(settings.SYSTEM_PROMPT_FILE).expanduser()
        if not path.is_absolute():
            cwd_path = Path.cwd() / path
            repo_root = Path(__file__).resolve().parents[2]
            repo_path = repo_root / path
            path = cwd_path if cwd_path.exists() else repo_path
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                f"Failed to read SYSTEM_PROMPT_FILE at '{path}': {exc}"
            ) from exc
        if not text:
            raise RuntimeError(f"SYSTEM_PROMPT_FILE is empty: '{path}'")
        return text

    if settings.SYSTEM_PROMPT:
        return settings.SYSTEM_PROMPT

    raise RuntimeError(
        "Missing SYSTEM_PROMPT configuration. Set SYSTEM_PROMPT_FILE "
        "or SYSTEM_PROMPT in local .env or runtime environment."
    )


SYSTEM_PROMPT = _load_system_prompt()


def build_user_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join(
        [
            f"SOURCE: {c['source']} | CHUNK: {c['chunk_id']}\n{c['text']}"
            for c in contexts
        ]
    )
    intent = detect_answer_intent(question)
    style = ""
    if intent == SUMMARY:
        style = (
            "STYLE:\n"
            "- Provide a clean summary format.\n"
            "- First line: one-sentence overview.\n"
            "- Add section header: Key Features.\n"
            "- Then provide 6-10 numbered items with 1-2 lines each.\n"
            "- Cover broad document scope; avoid spending more than 2 "
            "items on one subtopic.\n"
            "- Keep related items together so flow is continuous and "
            "easy to scan.\n"
            "- Each bullet must include at least one inline citation: "
            "[source | chunk_id].\n"
            "- Do not include uncertainty text unless context is truly "
            "insufficient.\n\n"
        )
    elif intent == DEFINITION:
        style = (
            "STYLE:\n"
            "- Start with a simple one- or two-sentence definition.\n"
            "- Use only facts grounded in the provided context.\n"
            "- If the context includes examples, add up to 2 brief examples.\n"
            "- Keep the answer compact and cite each factual statement.\n\n"
        )
    elif intent == EXAMPLES:
        style = (
            "STYLE:\n"
            "- Answer with brief examples found in the provided context.\n"
            "- If the topic needs a short setup sentence, keep it to one line.\n"
            "- Do not invent examples not supported by the context.\n"
            "- Cite each example inline.\n\n"
        )
    elif intent == COMPARISON:
        style = (
            "STYLE:\n"
            "- Compare the items directly and briefly.\n"
            "- Prefer 2-4 clear contrast points.\n"
            "- Cite each comparison point inline.\n\n"
        )
    elif intent == WHEN_TO_USE:
        style = (
            "STYLE:\n"
            "- Explain when the topic should be used based on the provided context.\n"
            "- Prefer short recommendation-style bullets or sentences.\n"
            "- If tradeoffs are present in the context, include them.\n"
            "- Cite each factual point inline.\n\n"
        )
    return f"{style}CONTEXT:\n{ctx}\n\nQUESTION:\n{question}"
