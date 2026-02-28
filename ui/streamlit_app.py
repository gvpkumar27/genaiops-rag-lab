import os
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
API_KEY = os.getenv("API_KEY", "").strip()
PUBLIC_UI_MODE = os.getenv("PUBLIC_UI_MODE", "true").lower() == "true"
SHOW_CITATION_DEBUG = (
    os.getenv("SHOW_CITATION_DEBUG", "false").lower() == "true"
)


def _new_chat() -> dict:
    chat_id = str(uuid4())[:8]
    return {"id": chat_id, "title": "New Chat", "messages": []}


def _init_state() -> None:
    if "chats" not in st.session_state:
        first = _new_chat()
        st.session_state.chats = [first]
        st.session_state.active_chat_id = first["id"]


def _active_chat() -> dict:
    active_id = st.session_state.active_chat_id
    for entry in st.session_state.chats:
        if entry["id"] == active_id:
            return entry
    fallback = st.session_state.chats[0]
    st.session_state.active_chat_id = fallback["id"]
    return fallback


def _chat_title_from_prompt(prompt: str) -> str:
    text = " ".join(prompt.strip().split())
    if not text:
        return "New Chat"
    return text[:45] + ("..." if len(text) > 45 else "")


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return

    if PUBLIC_UI_MODE and not SHOW_CITATION_DEBUG:
        source_to_label = {}
        source_counts = {}
        next_idx = 1
        for citation in citations:
            source = citation.get("source", "unknown")
            if source not in source_to_label:
                source_to_label[source] = f"Document {next_idx}"
                next_idx += 1
            source_counts[source] = source_counts.get(source, 0) + 1

        with st.expander("Sources"):
            for source, label in source_to_label.items():
                filename = Path(str(source)).name
                count = source_counts.get(source, 1)
                st.write(f"- {label}: {filename} ({count} citation hits)")
        return

    with st.expander("Sources (debug)"):
        for citation in citations:
            source = citation.get("source", "unknown")
            chunk_id = citation.get("chunk_id", -1)
            score = citation.get("score", 0.0)
            preview = citation.get("text_preview", "")
            st.markdown(f"**{source} | chunk {chunk_id} | score {score:.3f}**")
            st.write(preview)


def _render_meta(meta: dict) -> None:
    if not meta:
        return
    st.caption(
        f"retrieve={float(meta.get('retrieve_sec', 0.0)):.3f}s "
        f"rerank={float(meta.get('rerank_sec', 0.0)):.3f}s "
        f"generate={float(meta.get('generate_sec', 0.0)):.3f}s "
        f"cache_hit={bool(meta.get('cache_hit', False))} "
        f"fallback={bool(meta.get('fallback_used', False))}"
    )


def _chat_matches(entry: dict, query: str) -> bool:
    if not query:
        return True
    haystack = " ".join(
        [entry.get("title", "")]
        + [
            msg.get("content", "")
            for msg in entry.get("messages", [])
            if msg.get("role") == "user"
        ]
    ).lower()
    return query in haystack


def _filtered_chats(query: str) -> list[dict]:
    normalized = query.strip().lower()
    return [
        entry
        for entry in st.session_state.chats
        if _chat_matches(entry, normalized)
    ]


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Welcome")
        st.write(
            "Ask questions on your uploaded documents with grounded citations."
        )

        if st.button("New Chat", use_container_width=True):
            new_chat = _new_chat()
            st.session_state.chats.insert(0, new_chat)
            st.session_state.active_chat_id = new_chat["id"]
            st.rerun()

        query = st.text_input(
            "Search Chats",
            placeholder="Find by title or message",
        )

        st.subheader("Your Chats")
        filtered = _filtered_chats(query)

        if not filtered:
            st.caption("No chats found.")
            return

        for entry in filtered:
            active = entry["id"] == st.session_state.active_chat_id
            label = entry.get("title", "New Chat")
            prefix = "-> " if active else ""
            if st.button(
                f"{prefix}{label}",
                key=f"chat_{entry['id']}",
                use_container_width=True,
            ):
                st.session_state.active_chat_id = entry["id"]
                st.rerun()


def _request_answer(prompt: str) -> tuple[str, list[dict], dict]:
    headers = {"x-api-key": API_KEY} if API_KEY else None
    response = requests.post(
        API_URL,
        json={"question": prompt},
        headers=headers,
        timeout=600,
    )
    if response.status_code >= 400:
        error_msg = (
            "Sorry, I couldn't generate an answer right now. "
            "Please try again."
        )
        if response.status_code == 400:
            try:
                detail = response.json().get("detail", error_msg)
                error_msg = str(detail)
            except ValueError:
                error_msg = "Please enter a meaningful question."
        return error_msg, [], {}

    data = response.json()
    return data.get("answer", ""), data.get("citations", []), data.get("meta", {})


def _render_assistant_reply(prompt: str) -> tuple[str, list[dict], dict]:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                assistant_text, citations, meta = _request_answer(prompt)
                if citations or meta:
                    st.write(assistant_text)
                    _render_citations(citations)
                    _render_meta(meta)
                else:
                    st.error(assistant_text)
            except requests.RequestException:
                assistant_text = "Service is temporarily unavailable. Please try again."
                citations = []
                meta = {}
                st.error(assistant_text)
    return assistant_text, citations, meta


def main() -> None:
    _init_state()
    _render_sidebar()
    active_chat = _active_chat()

    st.title("Local AI Knowledge Companion")
    st.caption("Grounded answers from your local documents.")

    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                _render_citations(message.get("citations", []))
                _render_meta(message.get("meta", {}))

    prompt = st.chat_input("Ask me anything about your documents")
    if not prompt:
        return
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a non-empty question.")
        return

    if active_chat["title"] == "New Chat":
        active_chat["title"] = _chat_title_from_prompt(prompt)

    active_chat["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    assistant_text, citations, meta = _render_assistant_reply(prompt)

    active_chat["messages"].append(
        {
            "role": "assistant",
            "content": assistant_text,
            "citations": citations,
            "meta": meta,
        }
    )


main()
