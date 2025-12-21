import os
import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="SailRAG", layout="wide")
st.title("SailRAG — Yacht & Vessel Knowledge Base")

st.write("This UI will evolve into Upload → Search → Chat (RAG). For now: connectivity check.")

if st.button("Ping backend /healthz"):
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(f"{BACKEND_URL}/healthz")
        st.success("Backend responded")
        st.json(r.json())
    except Exception as e:
        st.error(f"Failed to reach backend: {e}")
