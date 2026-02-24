from app.config import settings
import re

SYSTEM_PROMPT = f"""
You are LocalDocChat.
Answer ONLY using the provided context.
Use semantically equivalent terms from context when wording differs from the question.
If context is insufficient, say exactly: "I don't know based on the provided documents."
Every factual sentence must be grounded in context and include inline citation(s) like: [source | chunk_id]
Never invent entities, numbers, dates, APIs, commands, versions, or causes that are not explicitly supported by context.
Do not use world knowledge when context is missing; return the exact unknown sentence instead.
Do not mix the unknown sentence with any additional explanation or bullets.
PROMPT_VERSION={settings.PROMPT_VERSION}
""".strip()

def build_user_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join([f"SOURCE: {c['source']} | CHUNK: {c['chunk_id']}\n{c['text']}" for c in contexts])
    q_low = question.lower()
    is_summary = bool(re.search(r"\b(summarize|summarise|summary|overview|highlights|features)\b", q_low))
    style = ""
    if is_summary:
        style = (
            "STYLE:\n"
            "- Provide a clean summary format.\n"
            "- First line: one-sentence overview.\n"
            "- Add section header: Key Features.\n"
            "- Then provide 6-10 numbered items with 1-2 lines each.\n"
            "- Cover broad document scope; avoid spending more than 2 items on one subtopic.\n"
            "- Keep related items together so flow is continuous and easy to scan.\n"
            "- Each bullet must include at least one inline citation: [source | chunk_id].\n"
            "- Do not include uncertainty text unless context is truly insufficient.\n\n"
        )
    return f"{style}CONTEXT:\n{ctx}\n\nQUESTION:\n{question}"
