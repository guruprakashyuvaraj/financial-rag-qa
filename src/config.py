"""
Centralized configuration loader for the Financial Document Q&A RAG system.
All environment-driven settings live here so the rest of the codebase
never touches os.environ directly.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # LLM: "openai" | "bedrock" | "ollama" (ollama = free, local, no API key)
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # Embeddings: "openai" | "bedrock" | "huggingface" (huggingface = free, local, no API key)
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # AWS Bedrock
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    bedrock_model_id: str = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
    bedrock_embedding_model_id: str = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v1")

    # Ollama (free, local LLM)
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # HuggingFace (free, local embeddings)
    hf_embedding_model: str = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Vector store
    vector_store: str = os.getenv("VECTOR_STORE", "faiss")
    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", "./data/faiss_index")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "financial-docs-qa")

    # Chunking
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "recursive")  # "recursive" | "semantic"
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
