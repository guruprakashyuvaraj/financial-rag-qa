"""
Vector store abstraction over FAISS (local) and Pinecone (cloud), so the
RAG pipeline can switch backends via a single config flag.
"""
import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.config import settings


class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.backend = settings.vector_store
        self._store = None

    # ---------- Build / Load ----------

    def build_from_documents(self, chunks: List[Document]):
        if self.backend == "pinecone":
            self._store = self._build_pinecone(chunks)
        else:
            self._store = self._build_faiss(chunks)
        return self._store

    def _build_faiss(self, chunks: List[Document]):
        store = FAISS.from_documents(chunks, self.embeddings)
        os.makedirs(settings.faiss_index_dir, exist_ok=True)
        store.save_local(settings.faiss_index_dir)
        return store

    def _build_pinecone(self, chunks: List[Document]):
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore

        pc = Pinecone(api_key=settings.pinecone_api_key)
        existing = [idx["name"] for idx in pc.list_indexes()]

        if settings.pinecone_index_name not in existing:
            # Dimension depends on embedding model; 1536 fits OpenAI's
            # text-embedding-3-small / ada-002 family.
            pc.create_index(
                name=settings.pinecone_index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=settings.pinecone_environment),
            )

        return PineconeVectorStore.from_documents(
            chunks,
            embedding=self.embeddings,
            index_name=settings.pinecone_index_name,
        )

    def load_existing(self) -> Optional[object]:
        if self.backend == "faiss" and os.path.exists(settings.faiss_index_dir):
            self._store = FAISS.load_local(
                settings.faiss_index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            return self._store

        if self.backend == "pinecone":
            from langchain_pinecone import PineconeVectorStore

            self._store = PineconeVectorStore(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings,
            )
            return self._store

        return None

    # ---------- Retrieval ----------

    def as_retriever(self, k: Optional[int] = None):
        if self._store is None:
            raise RuntimeError("Vector store not built or loaded yet.")
        return self._store.as_retriever(search_kwargs={"k": k or settings.top_k})
