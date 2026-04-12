"""
RAG Pipeline for Agency Knowledge Base

Loads client documents from data/, chunks them, embeds with
sentence-transformers, and stores in ChromaDB for retrieval.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .embeddings import get_embeddings
from .indexing import index_documents
from .vectorstore import create_vectorstore, load_vectorstore


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "vectorstore"
COLLECTION_NAME = "agency_knowledge_base"

# Embedding model — multilingual, runs locally, no API key needed
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def query(question: str, k: int = 4) -> list:
    """
    Query the vector store and return relevant document chunks.
    Useful for testing the retrieval without an LLM.
    """
    embeddings = get_embeddings(EMBEDDING_MODEL)
    vectorstore = load_vectorstore(
        embeddings,
        str(CHROMA_DIR),
        COLLECTION_NAME,
    )
    results = vectorstore.similarity_search(question, k=k)
    return results


if __name__ == "__main__":
    index_documents(
        DATA_DIR,
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_MODEL,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

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
