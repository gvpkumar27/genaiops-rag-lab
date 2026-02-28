import re
from pathlib import Path

from app.config import settings


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
    q_low = question.lower()
    is_summary = bool(
        re.search(
            r"\b(summarize|summarise|summary|overview|highlights|features)\b",
            q_low,
        )
    )
    style = ""
    if is_summary:
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
    return f"{style}CONTEXT:\n{ctx}\n\nQUESTION:\n{question}"
