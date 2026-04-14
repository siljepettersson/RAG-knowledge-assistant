import re
from pathlib import Path

from .embeddings import get_embeddings
from .vectorstore import load_vectorstore


def prepare_query_for_embedding(question: str, max_query_length: int) -> str:
    """Normalize and cap query text before embedding."""
    cleaned_question = question.replace("\u00a0", " ")
    cleaned_question = re.sub(r"\s+", " ", cleaned_question).strip()

    if not cleaned_question:
        raise ValueError("Query cannot be empty.")

    if max_query_length < 1:
        raise ValueError("max_query_length must be at least 1.")

    if len(cleaned_question) > max_query_length:
        cleaned_question = cleaned_question[:max_query_length].rstrip()

    return cleaned_question


def query(
    question: str,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    embedding_device: str = "cpu",
    normalize_embeddings: bool = True,
    embedding_batch_size: int = 32,
    max_query_length: int = 1000,
    k: int = 4,
) -> list:
    """Query the vector store and return relevant document chunks."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    prepared_question = prepare_query_for_embedding(question, max_query_length)

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
    return vectorstore.similarity_search(prepared_question, k=k)
