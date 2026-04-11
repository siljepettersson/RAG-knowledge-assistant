"""
RAG Pipeline for Agency Knowledge Base

Loads client documents from data/, chunks them, embeds with
sentence-transformers, and stores in ChromaDB for retrieval.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "vectorstore"

# Embedding model — multilingual, runs locally, no API key needed
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents() -> list:
    """Load all markdown documents from data/ directory."""
    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    # Add client name as metadata based on subfolder
    for doc in docs:
        path = Path(doc.metadata["source"])
        # Get client folder name (e.g. "fjordmat", "spareklar")
        relative = path.relative_to(DATA_DIR)
        client = relative.parts[0]
        doc.metadata["client"] = client
        doc.metadata["filename"] = relative.name

    return docs


def chunk_documents(docs: list) -> list:
    """Split documents into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """Initialize the embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


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
    docs = load_documents()
    print(f"  Loaded {len(docs)} documents")

    print("Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks")

    print("Initializing embedding model...")
    embeddings = get_embeddings()

    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks, embeddings)
    print(f"  Stored {vectorstore._collection.count()} vectors in {CHROMA_DIR}")

    return vectorstore


def query(question: str, k: int = 4) -> list:
    """
    Query the vector store and return relevant document chunks.
    Useful for testing the retrieval without an LLM.
    """
    embeddings = get_embeddings()
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
