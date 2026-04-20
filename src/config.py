from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


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
    batch_size: int = 32


@dataclass
class RetrievalConfig:
    collection_name: str = "agency_knowledge_base"
    top_k: int = 4
    max_query_length: int = 1000


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "openai_compatible")
    model_name: str = os.getenv("LLM_MODEL", "your-llm-model")
    base_url: str = os.getenv("LLM_BASE_URL", "")
    api_key: str = os.getenv("LLM_API_KEY", "")
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass(frozen=True)
class LLMProviderChoice:
    label: str
    config: LLMConfig


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


def get_available_llm_choices() -> list[LLMProviderChoice]:
    """Return LLM choices that have API keys configured in the environment."""
    choices: list[LLMProviderChoice] = []

    if os.getenv("LLM_API_KEY", "").strip():
        provider = os.getenv("LLM_PROVIDER", "openai_compatible")
        model_name = os.getenv("LLM_MODEL", "your-llm-model")
        choices.append(
            LLMProviderChoice(
                label=os.getenv("LLM_LABEL", f"Custom {provider} ({model_name})"),
                config=LLMConfig(
                    provider=provider,
                    model_name=model_name,
                    base_url=os.getenv("LLM_BASE_URL", ""),
                    api_key=os.getenv("LLM_API_KEY", ""),
                ),
            )
        )

    if os.getenv("OPENAI_API_KEY", "").strip():
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        choices.append(
            LLMProviderChoice(
                label=f"OpenAI ({model_name})",
                config=LLMConfig(
                    provider="openai_compatible",
                    model_name=model_name,
                    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                ),
            )
        )

    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        choices.append(
            LLMProviderChoice(
                label=f"Anthropic ({model_name})",
                config=LLMConfig(
                    provider="anthropic",
                    model_name=model_name,
                    base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
                    api_key=os.getenv("ANTHROPIC_API_KEY", ""),
                ),
            )
        )

    return choices


config = AppConfig()
