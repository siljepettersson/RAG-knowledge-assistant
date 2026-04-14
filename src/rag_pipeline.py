"""
RAG Pipeline for Agency Knowledge Base

Loads client documents from data/, chunks them, embeds with
sentence-transformers, and stores in ChromaDB for retrieval.
"""

from .config import config
from .indexing import index_documents
from .query import query


if __name__ == "__main__":
    index_documents(
        config.paths.data_dir,
        config.paths.vectorstore_dir,
        config.retrieval.collection_name,
        config.embedding.model_name,
        config.embedding.device,
        config.embedding.normalize_embeddings,
        config.chunking.chunk_size,
        config.chunking.chunk_overlap,
        config.chunking.min_chunk_length,
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
        results = query(
            question,
            config.paths.vectorstore_dir,
            config.retrieval.collection_name,
            config.embedding.model_name,
            embedding_device=config.embedding.device,
            normalize_embeddings=config.embedding.normalize_embeddings,
            k=2,
        )
        print(f"\nQ: {question}")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata['client']}/{doc.metadata['filename']}")
            print(f"      {doc.page_content[:150].strip()}...")
