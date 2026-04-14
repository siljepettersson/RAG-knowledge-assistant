from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=8)
def _build_embeddings(
    model_name: str,
    device: str,
    normalize_embeddings: bool,
    batch_size: int,
) -> HuggingFaceEmbeddings:
    """Create and cache embedding clients by configuration."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={
            "normalize_embeddings": normalize_embeddings,
            "batch_size": batch_size,
        },
    )


def get_embeddings(
    model_name: str,
    device: str = "cpu",
    normalize_embeddings: bool = True,
    batch_size: int = 32,
) -> HuggingFaceEmbeddings:
    """Return a cached embedding client for the requested configuration."""
    cleaned_model_name = model_name.strip()
    cleaned_device = device.strip().lower()

    if not cleaned_model_name:
        raise ValueError("Embedding model name cannot be empty.")

    if batch_size < 1:
        raise ValueError("Embedding batch_size must be at least 1.")

    return _build_embeddings(
        cleaned_model_name,
        cleaned_device,
        normalize_embeddings,
        batch_size,
    )
