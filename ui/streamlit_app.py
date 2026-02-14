import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

st.title("LocalDocChat (Open-source LLM + RAG)")
q = st.text_input("Ask a question about your documents:")

if st.button("Ask") and q.strip():
    try:
        r = requests.post(API_URL, json={"question": q}, timeout=600)
        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.json().get("detail", "")
            except ValueError:
                detail = r.text
            st.error(f"Chat request failed ({r.status_code}): {detail or 'Unknown error'}")
            st.stop()

        data = r.json()

        st.subheader("Answer")
        st.write(data["answer"])

        st.subheader("Citations")
        for c in data["citations"]:
            st.write(f"- **{c['source']} | chunk {c['chunk_id']}** (score: {c['score']:.3f})")
            st.caption(c["text_preview"])
    except requests.RequestException as e:
        st.error(f"Could not connect to API at {API_URL}: {e}")

