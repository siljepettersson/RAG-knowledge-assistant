"""
Phase 2 demo entrypoint for the agency knowledge base pipeline.

Running this module rebuilds the vector store from the current data directory
and then executes a small set of demo retrieval queries.
"""

from .config import config
from .indexing import index_documents
from .query import query

DEMO_QUESTIONS = [
    "Hva er Fjordmats tone of voice?",
    "Hva var ROAS for Spareklars Google Ads i Q4 2024?",
    "Hvilke influencere samarbeider Nordvik med?",
    "Hvor mange ansatte har LogistikkPartner?",
]


def run_indexing() -> None:
    """Rebuild the vector store from the current project documents."""
    index_documents(
        config.paths.data_dir,
        config.paths.vectorstore_dir,
        config.retrieval.collection_name,
        config.embedding.model_name,
        config.embedding.device,
        config.embedding.normalize_embeddings,
        config.embedding.batch_size,
        config.chunking.chunk_size,
        config.chunking.chunk_overlap,
        config.chunking.min_chunk_length,
    )


def run_demo_queries() -> None:
    """Run example retrieval queries against the rebuilt vector store."""
    print("\n--- Test Queries ---")
    for question in DEMO_QUESTIONS:
        results = query(
            question,
            config.paths.vectorstore_dir,
            config.retrieval.collection_name,
            config.embedding.model_name,
            embedding_device=config.embedding.device,
            normalize_embeddings=config.embedding.normalize_embeddings,
            embedding_batch_size=config.embedding.batch_size,
            max_query_length=config.retrieval.max_query_length,
            k=2,
        )
        print(f"\nQ: {question}")
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata['client']}/{doc.metadata['filename']}")
            print(f"      {doc.page_content[:150].strip()}...")


def main() -> None:
    """Run the Phase 2 rebuild-and-query demo flow."""
    run_indexing()
    run_demo_queries()


if __name__ == "__main__":
    main()
