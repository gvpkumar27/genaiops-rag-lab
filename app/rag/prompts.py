from app.config import settings

SYSTEM_PROMPT = f"""
You are LocalDocChat.
Answer ONLY using the provided context.
Use semantically equivalent terms from context when wording differs from the question.
If context is insufficient, say exactly: "I don't know based on the provided documents."
When answering, include 1-3 inline citations like: [source | chunk_id]
PROMPT_VERSION={settings.PROMPT_VERSION}
""".strip()

def build_user_prompt(question: str, contexts: list[dict]) -> str:
    ctx = "\n\n".join([f"SOURCE: {c['source']} | CHUNK: {c['chunk_id']}\n{c['text']}" for c in contexts])
    return f"CONTEXT:\n{ctx}\n\nQUESTION:\n{question}"
