import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vectorstore(
    chunks: list,
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str,
    collection_name: str,
) -> Chroma:
    """Create and persist a ChromaDB vector store from document chunks."""
    chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    return Chroma.from_documents(
        documents=chunks,
        ids=chunk_ids,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def rebuild_vectorstore(
    chunks: list,
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str,
    collection_name: str,
) -> Chroma:
    """Rebuild the persisted Chroma store from a clean directory."""
    persist_path = Path(persist_directory)

    if persist_path.exists():
        shutil.rmtree(persist_path)

    persist_path.mkdir(parents=True, exist_ok=True)

    return create_vectorstore(
        chunks,
        embeddings,
        persist_directory,
        collection_name,
    )


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str,
    collection_name: str,
) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
