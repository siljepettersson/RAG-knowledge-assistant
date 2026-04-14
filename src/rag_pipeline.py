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


def format_source_label(doc, source_number: int) -> str:
    """Build a readable source label for prompt-ready context formatting."""
    chunk_id = doc.metadata.get("chunk_id")

    if not chunk_id:
        client = doc.metadata["client"]
        filename = doc.metadata["filename"]
        chunk_index = doc.metadata["chunk_index"]
        chunk_id = f"{client}/{filename}#chunk-{chunk_index}"

    return f"[Source {source_number}] {chunk_id}"


def format_retrieved_context(retrieved_docs: list) -> str:
    """Format retrieved chunks into a stable, prompt-ready context block."""
    sections: list[str] = []

    for i, doc in enumerate(retrieved_docs, 1):
        source_label = format_source_label(doc, i)
        sections.append(f"{source_label}\n{doc.page_content.strip()}")

    return "\n\n".join(sections)


def format_source_list(retrieved_docs: list) -> list[str]:
    """Return a compact source list for logging and later answer generation."""
    return [format_source_label(doc, i) for i, doc in enumerate(retrieved_docs, 1)]


def build_prompt(question: str, context_block: str) -> str:
    """Build a prompt-ready RAG input without calling an LLM."""
    return (
        "You are an internal knowledge assistant for a marketing agency.\n"
        "Answer the user's question using only the provided context.\n"
        "If the context does not contain enough information, say that clearly.\n"
        "Reference the relevant source labels in your answer.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block}"
    )


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
        sources = format_source_list(results)
        context_block = format_retrieved_context(results)
        prompt = build_prompt(question, context_block)

        print(f"\nQ: {question}")
        print("Sources:")
        for source in sources:
            print(f"  - {source}")
        print("Context:")
        print(context_block)
        print("Prompt:")
        print(prompt)


def main() -> None:
    """Run the Phase 2 rebuild-and-query demo flow."""
    run_indexing()
    run_demo_queries()


if __name__ == "__main__":
    main()
