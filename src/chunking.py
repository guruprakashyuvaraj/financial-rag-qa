"""
Chunking strategies for converting raw financial documents into
retrieval-friendly chunks.

Supports:
- recursive: fast, structure-aware character/token splitting (default)
- semantic: embedding-based splitting that groups semantically similar
  sentences together, useful for dense financial narrative text
"""
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def recursive_chunk(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def semantic_chunk(documents: List[Document], embeddings) -> List[Document]:
    """
    Semantic chunking groups sentences by embedding similarity rather than
    a fixed character count. Falls back to recursive chunking if the
    experimental splitter isn't available.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker

        splitter = SemanticChunker(embeddings)
        return splitter.split_documents(documents)
    except Exception as exc:
        print(f"[chunking] Semantic chunking unavailable ({exc}); falling back to recursive.")
        return recursive_chunk(documents)


def chunk_documents(documents: List[Document], embeddings=None) -> List[Document]:
    """Dispatch to the configured chunking strategy."""
    if settings.chunk_strategy == "semantic" and embeddings is not None:
        return semantic_chunk(documents, embeddings)
    return recursive_chunk(documents)
