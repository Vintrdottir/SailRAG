from __future__ import annotations


def build_rag_prompt(question: str, contexts: list[dict]) -> str:
    """
    contexts: list of dicts with keys: doc_id, page_number, chunk_id, text
    """
    ctx_blocks = []
    for i, c in enumerate(contexts, start=1):
        header = f"[{i}] doc={c['doc_id']} page={c['page_number']} chunk={c['chunk_id']}"
        ctx_blocks.append(f"{header}\n{c['text']}".strip())

    joined = "\n\n---\n\n".join(ctx_blocks)

    return f"""You are a helpful assistant for recreational sailing and navigation.
Answer the question using ONLY the provided context.
If the context is insufficient, say what is missing and ask a precise follow-up question.

Question:
{question}

Context:
{joined}

Write a clear, practical answer in English. Include short bullet points when helpful.
"""
