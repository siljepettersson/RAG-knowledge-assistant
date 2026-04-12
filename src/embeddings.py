from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(model_name=model_name)
