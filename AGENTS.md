# AGENTS.md — Project Guide for AI Assistants

## What is this project?

This project started as a **demo/MVP**, but the current direction is to evolve it into a stronger RAG assistant demo with a real UI, LLM-generated answers, source citations, retrieval trace, and clear extension points for advanced retrieval.

The challenge:
1. Describe what you'd prioritize building first as an AI developer at the agency, and why
2. Build a simple MVP/demo/POC showing the idea in practice
3. Write about approach, choices made, what you'd do with more time, and time spent

**The idea:** A RAG-based Client Knowledge Base — an internal chatbot where agency employees can ask questions about any client's brand guidelines, campaigns, reports, and strategies, and get accurate answers with source references.

## Why this idea?

A marketing agency manages dozens of clients, each with brand guidelines, campaign briefs, performance reports, and meeting notes. New employees, account managers, and creatives constantly need to look up client-specific information. A RAG chatbot makes this knowledge instantly accessible.

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Framework | LangChain | Industry standard for RAG |
| Vector DB | ChromaDB | Simple (pip install, no server), good for demos |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 | Runs locally, free, good Norwegian support |
| UI | Streamlit | Quick to build, clean chat interface, easy to demo |
| LLM | TBD | Needed for the Streamlit app (Phase 3) |
| Language | Python | Everything in Python |

**Key design decision:** No API keys required for the core pipeline (embeddings + vector store run locally). This means anyone can clone and test the retrieval without paying for anything. The LLM is the only component that may need an API key.

## Project Structure

```
RAG-knowledge-assistant/
├── AGENTS.md              # This file — project context for AI assistants
├── README.md              # Project usage notes and write-up
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency set for uv
├── requirements.txt       # Compatibility dependency list
├── .gitignore             # Excludes envs/ and vectorstore/
├── data/                  # Fictional client documents (Norwegian, markdown)
│   ├── fjordmat/          # Restaurant chain (4 docs)
│   ├── spareklar/         # Fintech/spare-app (4 docs)
│   ├── nordvik/           # Sustainable clothing brand (4 docs)
│   └── skytjenester/      # B2B SaaS company (4 docs)
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Loads markdown documents and attaches metadata
│   ├── chunking.py        # Splits documents into retrieval chunks
│   ├── embeddings.py      # Creates the embedding model
│   ├── vectorstore.py     # Chroma create/load operations
│   ├── indexing.py        # Orchestration for indexing documents into Chroma
│   ├── query.py           # Retrieval/query logic against the vector store
│   └── rag_pipeline.py    # Compatibility entry point that wires indexing + query
├── vectorstore/           # Generated persisted Chroma data (gitignored)
└── app.py                 # Streamlit UI (not built in this repo state)
```

## The Fictional Clients

Four fictional Norwegian companies with 4 documents each (16 total). Document types include brand guidelines, campaign briefs, performance reports, meeting notes, strategies, and customer cases.

| Client | Type | Documents |
|--------|------|-----------|
| **Fjordmat** | Restaurant chain | Brand guidelines, summer campaign brief, christmas campaign report, Q1 meeting notes |
| **Spareklar** | Fintech app | Brand guidelines, feature launch brief, Google Ads Q4 report, social media strategy |
| **Nordvik** | Sustainable clothing | Brand guidelines, autumn collection brief, influencer strategy, Instagram campaign report |
| **Skytjenester AS** | B2B SaaS | Brand guidelines, lead gen brief, SEO report, customer case study |

All documents are written in Norwegian and contain realistic details — budgets, KPIs, metrics, timelines, contact persons, and action items.

## Build Phases

### Phase 1: Client Data ✅ (PR #1)
- 16 fictional Norwegian client documents in `data/`

### Phase 2: RAG Pipeline ✅
- Load docs with LangChain `DirectoryLoader`
- Chunk with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
- Embed with sentence-transformers (multilingual model)
- Store in ChromaDB
- Refactored into dedicated modules under `src/`
- Tested: the indexing flow builds the vector store and the sample queries retrieve relevant documents

### Phase 3: LLM + Streamlit UI
- Chat interface where users ask questions about clients
- LLM generates answers based on retrieved chunks
- Show source documents with each answer
- Show retrieval trace so the demo explains how the answer was grounded
- Client filter in sidebar
- Keep `app.py` thin: UI only, no direct Chroma/vectorstore/prompt orchestration logic
- Add structured response objects so retrieval and answer generation can evolve without changing the UI contract
- **LLM choice not yet decided** — options: OpenAI-compatible API, Anthropic, or local model

### Phase 3 Architecture Rules
- `app.py` should only handle Streamlit layout, session state, user input, and display.
- Do not put vector store loading, embedding setup, prompt construction, or LLM provider details directly in `app.py`.
- Put high-level orchestration in a dedicated module such as `src/assistant.py`.
- Put structured dataclasses or response schemas in a dedicated module such as `src/schemas.py`.
- Keep low-level retrieval in `src/query.py` or a retrieval-specific module.
- Keep prompt/source formatting utilities reusable from `src/rag_pipeline.py` unless they need to be split later.
- Put LLM API handling in a dedicated module such as `src/llm.py`.

`src/assistant.py` is the public assistant service boundary for UI callers. It should expose a small stable API such as `answer_question(...)` and coordinate retrieval, prompt building, LLM generation, and response-status mapping. It exists so `app.py` can stay focused on Streamlit display logic and does not need to know Chroma, embeddings, prompt internals, or provider-specific LLM details.

Do not put low-level retrieval algorithms or provider-specific HTTP request code in `src/assistant.py`; keep those in retrieval and LLM modules.

Recommended Phase 3 flow:

```text
app.py
    -> src.assistant.answer_question(...)
        -> retrieve context
        -> build prompt
        -> generate LLM answer
        -> return structured AssistantResponse
```

The retrieval result should not be only a raw `list[Document]` or a plain string. Use a structured object that can grow over time.

Initial retrieved-context fields:
- `question`
- `retrieved_chunks`
- `source_labels`
- `context_block`
- `client_filter_used`
- `retrieval_notes`

Initial assistant-response fields:
- `question`
- `status`
- `answer`
- `sources`
- `retrieved_context`
- `prompt`
- `model_name`
- `error`

Include response status from the first implementation so the UI can branch cleanly. Suggested statuses:
- `answered`: retrieval and LLM generation both succeeded
- `retrieval_only`: retrieval worked, but no LLM was configured, so the app shows retrieved context or a prompt-ready fallback
- `no_results`: retrieval returned no useful context
- `configuration_error`: required settings such as an LLM API key or base URL are missing or invalid
- `runtime_error`: an unexpected error happened during retrieval or generation

Response contract conventions:
- Normalize UI client filter values before low-level retrieval. Retrieval functions should receive either `None` or a real client slug such as `fjordmat`; never pass display labels like `All clients`.
- `RetrievedContext.source_labels` are retrieval-produced labels for retrieved chunks.
- `AssistantResponse.sources` is the final UI-facing source list for the answer. It can initially mirror `source_labels`, but later may contain only sources used in the final answer.
- `RetrievedContext.retrieved_chunks` may contain LangChain `Document` objects internally, but `app.py` should not depend on detailed `Document.metadata` structure.
- `AssistantResponse.prompt` should store the full prompt for trace/debug. The UI can hide it by default or show a preview.

Future fields can include:
- `rewritten_queries`
- `subqueries`
- `rerank_scores`
- `evidence_status`
- `retrieval_rounds`
- `confidence`

### Advanced Retrieval Direction
- The embedding model remains responsible for vector representations.
- The LLM should first be used for answer generation.
- Later, the LLM can act as a retrieval controller on top of vector search.
- Advanced retrieval should be added behind the assistant/retrieval boundary, not inside the UI.

Planned advanced retrieval capabilities:
- query understanding: infer client, document type, and answer type
- query rewriting: make vague user questions searchable
- multi-query retrieval: split complex questions into several searches
- multi-hop retrieval: gather evidence from multiple documents
- reranking: prioritize chunks that best support the answer
- evidence sufficiency checks: decide whether the context is enough before answering

### Phase 4: Write-up (last)
- README.md with: what was built, why, approach, choices, what to improve, time spent

## Development Conventions

- **Language:** Documents are in Norwegian. Code and comments in English.
- **Git workflow:** Never push to main. Always use feature branches and PRs.
- **Branch naming:** Prefer descriptive refactor/feature branches (e.g. `refactor/query-module`)
- **PR base:** Each PR builds on the previous one until merged.
- **Dependencies:** Managed with `uv`. Defined in `pyproject.toml`, locked in `uv.lock`. Run `uv sync` to install.
- **Python version:** `>=3.11` (see `pyproject.toml`)

## Running the project

```bash
# Setup (using uv for dependency management)
uv sync

# Run the modular pipeline entry point
uv run python -m src.rag_pipeline

# Run the app (Phase 3, when built)
uv run streamlit run app.py
```

Do not rely on plain `python3 -m src.rag_pipeline` in the VM unless dependencies have been installed into that interpreter. The working command in this repo is the `uv run ...` invocation above.

## What to keep in mind

- This is a demo project, but do not restrict the architecture to a throwaway MVP. Keep it clean, explainable, and ready for advanced retrieval.
- The audience is a hiring manager at a marketing agency — the demo should be impressive but explainable.
- Showing good RAG fundamentals still matters more than flashy features.
- Favor clear module boundaries over putting everything into `app.py`.
- Prefer structured return objects over loosely passing strings and raw document lists between layers.
