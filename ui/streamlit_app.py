import os
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/chat")
API_KEY = os.getenv("API_KEY", "").strip()
PUBLIC_UI_MODE = os.getenv("PUBLIC_UI_MODE", "true").lower() == "true"
SHOW_CITATION_DEBUG = os.getenv("SHOW_CITATION_DEBUG", "false").lower() == "true"

st.title("Local AI Knowledge Companion")
q = st.text_input("Ask me anything about your documents:")

if st.button("Submit") and q.strip():
    try:
        headers = {"x-api-key": API_KEY} if API_KEY else None
        r = requests.post(API_URL, json={"question": q}, headers=headers, timeout=600)
        if r.status_code >= 400:
            # End-user UI: keep backend details out of the page.
            st.error("Sorry, I couldn't generate an answer right now. Please try again.")
            st.stop()

        data = r.json()
        st.subheader("Answer")
        st.write(data.get("answer", ""))

        citations = data.get("citations", [])
        if citations:
            st.subheader("Citations")
            if PUBLIC_UI_MODE and not SHOW_CITATION_DEBUG:
                source_to_label = {}
                source_counts = {}
                next_idx = 1
                for c in citations:
                    source = c.get("source", "unknown")
                    if source not in source_to_label:
                        source_to_label[source] = f"Document {next_idx}"
                        next_idx += 1
                    source_counts[source] = source_counts.get(source, 0) + 1
                for source, label in source_to_label.items():
                    label = source_to_label[source]
                    filename = Path(str(source)).name
                    cnt = source_counts.get(source, 1)
                    st.write(f"- {label}: {filename} ({cnt} citation hits)")
            else:
                for c in citations:
                    source = c.get("source", "unknown")
                    chunk_id = c.get("chunk_id", -1)
                    score = c.get("score", 0.0)
                    preview = c.get("text_preview", "")
                    with st.expander(f"{source} | chunk {chunk_id} | score {score:.3f}"):
                        st.write(preview)

        meta = data.get("meta", {})
        if meta:
            st.caption(
                "retrieve={:.3f}s rerank={:.3f}s generate={:.3f}s cache_hit={} fallback={}".format(
                    float(meta.get("retrieve_sec", 0.0)),
                    float(meta.get("rerank_sec", 0.0)),
                    float(meta.get("generate_sec", 0.0)),
                    bool(meta.get("cache_hit", False)),
                    bool(meta.get("fallback_used", False)),
                )
            )
    except requests.RequestException:
        st.error("Service is temporarily unavailable. Please try again.")
