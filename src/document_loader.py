"""
Document extraction layer.
Handles loading financial PDFs (and plain text) into LangChain Document objects.
"""
from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(file_path: str) -> List[Document]:
    """
    Load a single file (PDF or TXT) into a list of LangChain Documents,
    one per page for PDFs.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()

    # Tag every page with the source filename for traceability in citations.
    for doc in docs:
        doc.metadata["source_file"] = path.name

    return docs


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load multiple files and flatten into a single list of Documents."""
    all_docs: List[Document] = []
    for fp in file_paths:
        try:
            all_docs.extend(load_document(fp))
        except Exception as exc:
            print(f"[document_loader] Failed to load {fp}: {exc}")
    return all_docs
