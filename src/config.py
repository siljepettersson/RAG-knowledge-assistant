from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class PathsConfig:
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = project_root / "data"
    vectorstore_dir: Path = project_root / "vectorstore"


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_length: int = 30


@dataclass
class EmbeddingConfig:
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    device: str = "cpu"
    normalize_embeddings: bool = True


@dataclass
class RetrievalConfig:
    collection_name: str = "agency_knowledge_base"
    top_k: int = 4


@dataclass
class LLMConfig:
    provider: str = "openai_compatible"
    model_name: str = "your-llm-model"
    base_url: str = os.getenv("LLM_BASE_URL", "")
    api_key: str = os.getenv("LLM_API_KEY", "")
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


config = AppConfig()
