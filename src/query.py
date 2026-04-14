from pathlib import Path

from .embeddings import get_embeddings
from .vectorstore import load_vectorstore


def query(
    question: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    embedding_device: str = "cpu",
    normalize_embeddings: bool = True,
    embedding_batch_size: int = 32,
    k: int = 4,
) -> list:
    """Query the vector store and return relevant document chunks."""
    embeddings = get_embeddings(
        embedding_model,
        device=embedding_device,
        normalize_embeddings=normalize_embeddings,
        batch_size=embedding_batch_size,
    )
    vectorstore = load_vectorstore(
        embeddings,
        str(chroma_dir),
        collection_name,
    )
    return vectorstore.similarity_search(question, k=k)
