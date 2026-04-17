from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document


ResponseStatus = Literal[
    "answered",
    "retrieval_only",
    "no_results",
    "configuration_error",
    "runtime_error",
]
"""Assistant response state used by the UI.

- answered: retrieval and LLM generation both succeeded.
- retrieval_only: retrieval worked, but no LLM was configured.
- no_results: retrieval executed but returned no useful context.
- configuration_error: required settings are missing or invalid.
- runtime_error: an unexpected error happened during retrieval or generation.
"""


@dataclass
class RetrievedContext:
    """Structured retrieval result for UI display and future retrieval upgrades.

    source_labels are labels produced by retrieval for retrieved chunks.
    retrieved_chunks can keep LangChain Document objects internally, but UI code
    should avoid depending on detailed Document.metadata structure.
    """

    question: str
    retrieved_chunks: list[Document]
    source_labels: list[str]
    context_block: str
    client_filter_used: str | None = None
    retrieval_notes: list[str] = field(default_factory=list)


@dataclass
class AssistantResponse:
    """Structured assistant result returned to the UI.

    sources is the final UI-facing source list for the answer. It may initially
    mirror RetrievedContext.source_labels, and later can contain only sources
    used by the final generated answer.

    prompt stores the full prompt for trace/debug. The UI can hide it by
    default or show a shortened preview.
    """

    question: str
    status: ResponseStatus
    answer: str
    sources: list[str]
    retrieved_context: RetrievedContext | None = None
    prompt: str | None = None
    model_name: str | None = None
    error: str | None = None
