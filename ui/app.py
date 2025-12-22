import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

st.set_page_config(page_title="SailRAG", page_icon="⛵", layout="wide")

st.title("⛵ SailRAG — Nautical Document Q&A")
st.caption("Hybrid retrieval (BM25 + vector) + local LLM (Ollama) + citations")

with st.sidebar:
    st.header("Retrieval settings")
    k = st.slider("Top-k chunks", min_value=1, max_value=20, value=5, step=1)
    w_bm25 = st.slider("BM25 weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    w_knn = st.slider("kNN weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    st.divider()
    st.write("Backend:", BACKEND_URL)

question = st.text_area(
    "Ask a question about the ingested nautical PDFs",
    value="How do nautical charts help you avoid hazards in unfamiliar waters?",
    height=90,
)

col1, col2 = st.columns([1, 1])

ask = st.button("Ask", type="primary", use_container_width=True)

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    payload = {"question": question, "k": k, "w_bm25": w_bm25, "w_knn": w_knn}

    try:
        with st.spinner("Retrieving and generating answer..."):
            r = requests.post(f"{BACKEND_URL}/answer", json=payload, timeout=180)
        if r.status_code != 200:
            st.error(f"Backend error ({r.status_code}): {r.text}")
            st.stop()

        data = r.json()
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")
        st.stop()

    with col1:
        st.subheader("Answer")
        st.write(data.get("answer", ""))

    with col2:
        st.subheader("Citations")
        citations = data.get("citations", [])
        if not citations:
            st.info("No citations returned.")
        else:
            for i, c in enumerate(citations, start=1):
                header = f"[{i}] {c.get('doc_id')} — page {c.get('page_number')} — {c.get('chunk_id')}"
                with st.expander(header, expanded=(i == 1)):
                    st.write(c.get("text", ""))
