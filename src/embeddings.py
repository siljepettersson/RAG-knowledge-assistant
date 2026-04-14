from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(
    model_name: str,
    device: str = "cpu",
    normalize_embeddings: bool = True,
) -> HuggingFaceEmbeddings:
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},
    )
