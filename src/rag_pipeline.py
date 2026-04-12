"""
RAG Pipeline for Agency Knowledge Base

Loads client documents from data/, chunks them, embeds with
sentence-transformers, and stores in ChromaDB for retrieval.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .chunking import chunk_documents
from .data_loader import load_documents
from .embeddings import get_embeddings


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "vectorstore"

# Embedding model — multilingual, runs locally, no API key needed
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def create_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Create and persist a ChromaDB vector store from document chunks."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="agency_knowledge_base",
    )
    return vectorstore


def load_vectorstore(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="agency_knowledge_base",
    )


def build_vectorstore() -> Chroma:
    """Full pipeline: load docs → chunk → embed → store. Returns the vector store."""
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"  Loaded {len(docs)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"  Created {len(chunks)} chunks")

    print("Initializing embedding model...")
    embeddings = get_embeddings(EMBEDDING_MODEL)

    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks, embeddings)
    print(f"  Stored {vectorstore._collection.count()} vectors in {CHROMA_DIR}")

    return vectorstore


def query(question: str, k: int = 4) -> list:
    """
    Query the vector store and return relevant document chunks.
    Useful for testing the retrieval without an LLM.
    """
    embeddings = get_embeddings(EMBEDDING_MODEL)
    vectorstore = load_vectorstore(embeddings)
    results = vectorstore.similarity_search(question, k=k)
    return results


if __name__ == "__main__":
    # Build the vector store
    vectorstore = build_vectorstore()

    # Test queries
    test_questions = [
        "Hva er Fjordmats tone of voice?",
        "Hva var ROAS for Spareklars Google Ads i Q4 2024?",
        "Hvilke influencere samarbeider Nordvik med?",
        "Hvor mange ansatte har LogistikkPartner?",
    ]

    print("\n--- Test Queries ---")
    for question in test_questions:
        results = query(question, k=2)
        print(f"\nQ: {question}")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata['client']}/{doc.metadata['filename']}")
            print(f"      {doc.page_content[:150].strip()}...")
