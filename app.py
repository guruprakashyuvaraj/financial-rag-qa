"""
Streamlit UI for the AI-Powered Financial Document Q&A System.

Run with:
    streamlit run app.py
"""
import os
import tempfile

import streamlit as st

from src.rag_pipeline import FinancialRAGPipeline

st.set_page_config(page_title="Financial Document Q&A", page_icon="📊", layout="wide")

st.title("📊 AI-Powered Financial Document Q&A System")
st.caption("RAG over financial PDFs · LangChain · FAISS/Pinecone · OpenAI/AWS Bedrock")

# ---------------- Session state ----------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = FinancialRAGPipeline()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed" not in st.session_state:
    st.session_state.indexed = False

pipeline = st.session_state.pipeline

# ---------------- Sidebar: document upload ----------------
with st.sidebar:
    st.header("📁 Upload Financial Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files (10-K, 10-Q, earnings reports, etc.)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("🔎 Process & Index Documents", disabled=not uploaded_files):
        with st.spinner("Extracting text, chunking, and building vector index..."):
            tmp_paths = []
            tmp_dir = tempfile.mkdtemp()
            for uf in uploaded_files:
                path = os.path.join(tmp_dir, uf.name)
                with open(path, "wb") as f:
                    f.write(uf.getbuffer())
                tmp_paths.append(path)

            try:
                num_chunks = pipeline.ingest(tmp_paths)
                st.session_state.indexed = True
                st.success(f"Indexed {len(tmp_paths)} document(s) into {num_chunks} chunks.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    if st.button("📂 Load Previously Saved Index"):
        with st.spinner("Loading existing vector index..."):
            try:
                if pipeline.load_existing_index():
                    st.session_state.indexed = True
                    st.success("Existing index loaded.")
                else:
                    st.warning("No existing index found.")
            except Exception as exc:
                st.error(f"Failed to load index: {exc}")

    st.divider()
    st.markdown(
        "**Config tip:** set `LLM_PROVIDER` and `VECTOR_STORE` in your `.env` "
        "to switch between OpenAI/Bedrock and FAISS/Pinecone."
    )

# ---------------- Main: chat interface ----------------
st.subheader("💬 Ask a question about your financial documents")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander("📎 Sources"):
                for s in turn["sources"]:
                    st.markdown(
                        f"**{s['source_file']}** (page {s['page']})\n\n> {s['snippet']}..."
                    )

question = st.chat_input(
    "e.g. What was total revenue in Q3 and how did it compare to the prior year?"
)

if question:
    if not st.session_state.indexed:
        st.warning("Please upload and index at least one document first.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    result = pipeline.ask(question)
                    st.markdown(result["answer"])
                    with st.expander("📎 Sources"):
                        for s in result["sources"]:
                            st.markdown(
                                f"**{s['source_file']}** (page {s['page']})\n\n> {s['snippet']}..."
                            )
                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                except Exception as exc:
                    st.error(f"Failed to answer: {exc}")
