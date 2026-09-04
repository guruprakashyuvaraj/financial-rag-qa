"""
End-to-end Retrieval-Augmented Generation pipeline for financial document Q&A.

Flow:
  documents -> chunk -> embed -> vector store -> retriever
  question -> retriever -> grounded prompt -> LLM -> answer + sources
"""
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.document_loader import load_documents
from src.chunking import chunk_documents
from src.embeddings import get_embeddings
from src.vector_store import VectorStoreManager
from src.llm import get_llm

SYSTEM_PROMPT = """You are a meticulous financial analyst assistant.
Answer the user's question using ONLY the context extracted from the
provided financial documents below. If the answer cannot be found in the
context, say clearly that the documents do not contain that information —
do not guess or use outside knowledge.

When you use a number (revenue, margin, ratio, date, etc.), quote it exactly
as it appears in the context.

Context:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "?")
        formatted.append(f"[Source {i} | {src} | page {page}]\n{d.page_content}")
    return "\n\n".join(formatted)


class FinancialRAGPipeline:
    """
    Owns the full lifecycle: ingest documents -> build index -> answer questions.
    """

    def __init__(self):
        self.embeddings = get_embeddings()
        self.llm = get_llm()
        self.vsm = VectorStoreManager(self.embeddings)
        self.retriever = None

    def ingest(self, file_paths: List[str]) -> int:
        """Load, chunk, embed, and index the given files. Returns chunk count."""
        raw_docs = load_documents(file_paths)
        if not raw_docs:
            raise ValueError("No documents could be loaded from the provided files.")

        chunks = chunk_documents(raw_docs, embeddings=self.embeddings)
        self.vsm.build_from_documents(chunks)
        self.retriever = self.vsm.as_retriever()
        return len(chunks)

    def load_existing_index(self) -> bool:
        store = self.vsm.load_existing()
        if store is not None:
            self.retriever = self.vsm.as_retriever()
            return True
        return False

    def ask(self, question: str) -> Dict[str, Any]:
        if self.retriever is None:
            raise RuntimeError("No index available. Ingest documents first.")

        retrieved_docs = self.retriever.invoke(question)
        context = _format_docs(retrieved_docs)

        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough()}
            | QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        sources = [
            {
                "source_file": d.metadata.get("source_file", "unknown"),
                "page": d.metadata.get("page", "?"),
                "snippet": d.page_content[:300],
            }
            for d in retrieved_docs
        ]

        return {"answer": answer, "sources": sources}
