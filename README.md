# AI-Powered Financial Document Q&A System (RAG)

A Retrieval-Augmented Generation application that answers natural-language
questions about financial PDF documents (10-Ks, 10-Qs, earnings reports,
prospectuses, etc.) using grounded, citation-backed responses.

## Tech Stack
- **Python** / **LangChain** — orchestration
- **FAISS** / **Pinecone** — vector similarity search (switchable)
- **OpenAI API** / **AWS Bedrock** — embeddings + LLM generation (switchable)
- **Streamlit** — interactive chat UI with document upload

## Features
- PDF/TXT ingestion with per-page metadata for traceable citations
- Recursive character chunking (default) or semantic (embedding-based) chunking
- Pluggable vector store: local FAISS index or managed Pinecone index
- Pluggable LLM/embedding provider: OpenAI or AWS Bedrock
- Retrieval-grounded prompting to minimize hallucination — the model is
  instructed to say "not found in documents" rather than guess
- Streamlit chat interface with expandable source citations per answer
- Persisted FAISS index so you don't have to re-embed on every restart

## Project Structure
```
financial-rag-qa/
├── app.py                   # Streamlit entrypoint
├── requirements.txt
├── .env.example
├── data/
│   └── faiss_index/         # persisted local vector index (created at runtime)
└── src/
    ├── config.py            # env-driven settings
    ├── document_loader.py   # PDF/TXT extraction
    ├── chunking.py          # recursive + semantic chunking strategies
    ├── embeddings.py        # OpenAI / Bedrock embedding factory
    ├── vector_store.py      # FAISS / Pinecone abstraction
    ├── llm.py                # OpenAI / Bedrock chat model factory
    └── rag_pipeline.py       # ingest() + ask() orchestration
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env with your API keys and provider choices
```

## Run
```bash
streamlit run app.py
```

Then, in the browser:
1. Upload one or more financial PDF/TXT files in the sidebar.
2. Click **Process & Index Documents**.
3. Ask questions in the chat box — each answer includes expandable
   source snippets showing exactly which page/document it came from.

## Switching Providers
Edit `.env`:
```bash
LLM_PROVIDER=openai        # or "bedrock"
VECTOR_STORE=faiss         # or "pinecone"
CHUNK_STRATEGY=recursive   # or "semantic"
```
No code changes required — the factories in `src/llm.py`,
`src/embeddings.py`, and `src/vector_store.py` read these flags at runtime.

## Notes
- Semantic chunking uses `langchain_experimental.SemanticChunker` and falls
  back to recursive chunking automatically if unavailable.
- Pinecone indexes are auto-created (serverless, cosine metric, dim=1536)
  if they don't already exist.
- FAISS indexes persist to `./data/faiss_index` and can be reloaded via the
  "Load Previously Saved Index" sidebar button without re-embedding.
