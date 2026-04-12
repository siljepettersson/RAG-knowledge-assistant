from pathlib import Path

from langchain_chroma import Chroma

from .chunking import chunk_documents
from .data_loader import load_documents
from .embeddings import get_embeddings
from .vectorstore import create_vectorstore


def index_documents(
    data_dir: Path,
    chroma_dir: Path,
    collection_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Chroma:
    """Full pipeline: load docs, chunk, embed, and store."""
    print("Loading documents...")
    docs = load_documents(data_dir)
    print(f"  Loaded {len(docs)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(docs, chunk_size, chunk_overlap)
    print(f"  Created {len(chunks)} chunks")

    print("Initializing embedding model...")
    embeddings = get_embeddings(embedding_model)

    print("Creating vector store...")
    vectorstore = create_vectorstore(
        chunks,
        embeddings,
        str(chroma_dir),
        collection_name,
    )
    print(f"  Stored {vectorstore._collection.count()} vectors in {chroma_dir}")

    return vectorstore
