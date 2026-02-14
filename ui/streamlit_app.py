import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

st.title("Local AI Knowledge Companion")
q = st.text_input("Ask me anything about your documents:")

if st.button("Submit") and q.strip():
    try:
        r = requests.post(API_URL, json={"question": q}, timeout=600)
        if r.status_code >= 400:
            # End-user UI: keep backend details out of the page.
            st.error("Sorry, I couldn't generate an answer right now. Please try again.")
            st.stop()

        data = r.json()
        st.write(data.get("answer", ""))
    except requests.RequestException:
        st.error("Service is temporarily unavailable. Please try again.")
