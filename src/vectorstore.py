from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vectorstore(
    chunks: list,
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str,
    collection_name: str,
) -> Chroma:
    """Create and persist a ChromaDB vector store from document chunks."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
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
