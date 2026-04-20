# RAG Knowledge Assistant

This repository is a RAG project for an internal client knowledge assistant at a marketing agency. Agency employees can search client material such as brand guidelines, campaign briefs, reports, and strategy documents and retrieve relevant source-backed context.

The project started as a compact proof of concept, but the current direction is to turn the baseline retrieval pipeline into a usable assistant with a Streamlit chat UI, LLM-generated answers, source citations, and an architecture that can later support more advanced retrieval behavior.

![RAG pipeline flow](assets/rag-pipeline-flow.png)

RAG flow: prepare knowledge documents, split them into chunks, convert those chunks into embeddings, store the vector representations with their text and metadata, retrieve the most relevant chunks for a user query, and use the retrieved context for answer generation.

The current baseline uses a local embedding model for vector search. The LLM will be added in the answer-generation layer first, not as a replacement for embeddings. Later, the LLM can also help with retrieval planning, query rewriting, reranking, and evidence checks.

## Workflow Explained

### 1. Background Preparation: Data Ingestion

**Step 1. Ingestion and Chunking**

Raw knowledge documents are loaded, cleaned, enriched with metadata, and split into smaller text chunks that can be retrieved later.

**Step 2. Embedding Model**

Each text chunk is converted into a vector representation that captures its semantic meaning.

**Step 3. Vector Database**

The chunk vectors are stored in the vector database together with their original text and metadata.

### 2. User Interaction: Retrieval and Generation

**Step 4. Semantic Search Query**

The user's question is converted into a query vector and compared against the stored chunk vectors to find the most relevant context.

**Retrieval Engine**

**Step 5. Augmented Generation**

The retrieved context and the user question are combined into a prompt that an LLM can use to generate a grounded answer.

### 3. Memory Buffer

Memory can support this in several ways:

- **Storage:** after each answer, the conversation or a summary of it can be stored in memory.
- **Feedback loop:** when the next question arrives, relevant memory can be retrieved and combined with the new question.
- **Query rewriting:** a vague follow-up like "How do I apply for it?" can be rewritten as "How do I apply for vacation?"
- **Looped retrieval:** the clarified query can then be used to retrieve better knowledge base context.

Common memory types include:

- **Short-term memory:** remembers the last few conversation turns to keep the current dialogue coherent.
- **Long-term memory:** stores persistent preferences or facts across sessions, such as a user's preferred answer style.

Memory is not implemented in this project yet, but it is part of the larger target architecture shown in the diagram.

The project currently delivers a solid Phase 2 retrieval pipeline and a pre-LLM orchestration layer. It can:

- load fictional client documents from `data/`
- split them into retrieval chunks
- generate embeddings locally
- rebuild a Chroma vector store from a clean state
- run retrieval queries against that store
- format retrieved chunks into prompt-ready context with stable source labels

It now includes the Phase 3 assistant boundary and a thin Streamlit UI. If no LLM is configured, the app still runs in retrieval-only mode and shows sources, retrieved context, and the prompt that would be sent to an LLM.

## Phase 3 Direction

The next phase is to add the user-facing assistant layer while keeping the code ready for more advanced retrieval later.

The main design rule for Phase 3:

- `app.py` should stay thin and only handle UI concerns
- retrieval, prompt construction, LLM calls, source formatting, and response objects should live in `src/`
- the app should display answers, source citations, and a retrieval trace, but it should not know Chroma or embedding details

Planned Phase 3 structure:

```text
app.py
    ↓
src.assistant.answer_question(...)
    ↓
retrieve context
    ↓
build prompt
    ↓
generate LLM answer
    ↓
return structured response
```

`src/assistant.py` is the public assistant service boundary for the app. Its purpose is to expose a small, stable function such as `answer_question(...)` that coordinates retrieval, prompt building, LLM generation, and response-status mapping. This keeps `app.py` focused on Streamlit UI work and prevents UI code from depending on Chroma, embeddings, prompt internals, or provider-specific LLM details.

`src/assistant.py` should not become a place for low-level retrieval algorithms or provider-specific HTTP logic. Those responsibilities stay in modules such as `src/query.py`, `src/rag_pipeline.py`, and `src/llm.py`.

The retrieval result should be structured from the beginning instead of returning only raw documents or a plain context string. The first version can include:

- `question`
- `retrieved_chunks`
- `source_labels`
- `context_block`
- `client_filter_used`
- `retrieval_notes`

The assistant response can include:

- `question`
- `status`
- `answer`
- `sources`
- `retrieved_context`
- `prompt`
- `model_name`
- `error`

The `status` field should be included from the first version so the UI can decide how to present the result. Expected initial statuses:

- `answered`: retrieval and LLM generation both succeeded
- `retrieval_only`: retrieval worked, but no LLM was configured, so the app shows retrieved context or a prompt-ready fallback
- `no_results`: retrieval returned no useful context
- `configuration_error`: required settings such as an LLM API key or base URL are missing or invalid
- `runtime_error`: an unexpected error happened during retrieval or generation

Response contract conventions:

- UI display values such as `All clients` must be normalized before reaching low-level retrieval. Retrieval functions should receive either `None` or a real client slug such as `fjordmat`.
- `RetrievedContext.source_labels` means labels produced by retrieval for the retrieved chunks.
- `AssistantResponse.sources` means the final UI-facing source list for the answer. In the first version it may match `RetrievedContext.source_labels`; later it may contain only sources actually cited by the generated answer.
- `RetrievedContext.retrieved_chunks` may keep LangChain `Document` objects internally, but `app.py` should not depend on detailed `Document.metadata` structure.
- `AssistantResponse.prompt` stores the full prompt for trace/debug use. The UI can choose to hide it in an expander or show a shortened preview.

This gives the app a stable data contract. Future retrieval upgrades can add fields such as:

- `rewritten_queries`
- `subqueries`
- `rerank_scores`
- `evidence_status`
- `retrieval_rounds`
- `confidence`

## Future Advanced Retrieval Direction

After Phase 3 is working, the baseline retrieval can be upgraded without rewriting the UI.

Potential advanced retrieval layers:

- query understanding: identify client, document type, and answer type
- query rewriting: turn vague questions into searchable questions
- multi-query retrieval: split complex questions into several searches
- multi-hop retrieval: gather evidence from multiple related documents
- reranking: prioritize the chunks that best support the answer
- evidence sufficiency checks: decide whether the retrieved context is enough

The LLM should not replace the embedding model. Instead, it can act as a retrieval controller on top of the local vector search:

```text
User question
    ↓
Question understanding
    ↓
Query planning
    ↓
Vector retrieval
    ↓
Reranking / evidence selection
    ↓
Evidence sufficiency check
    ↓
Answer generation with citations
```

## Current Status

Implemented now:

- modular RAG pipeline under `src/`
- rebuild-style indexing pipeline
- stable `chunk_id` identity from chunking to storage to display
- query input guardrails
- prompt-ready context orchestration in `src.rag_pipeline`
- structured response objects in `src.schemas`
- LLM provider boundary in `src.llm`
- assistant orchestration boundary in `src.assistant`
- Streamlit chat UI in `app.py`

Not implemented yet:

- production-ready LLM configuration examples
- hybrid local + web retrieval

## Project Structure

```text
RAG-knowledge-assistant/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── app.py
├── data/
│   ├── fjordmat/
│   ├── nordvik/
│   ├── skytjenester/
│   └── spareklar/
├── src/
│   ├── __init__.py
│   ├── assistant.py
│   ├── chunking.py
│   ├── config.py
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── indexing.py
│   ├── llm.py
│   ├── query.py
│   ├── rag_pipeline.py
│   ├── schemas.py
│   └── vectorstore.py
└── vectorstore/
```

## Data

The repository contains 16 fictional Norwegian client documents across 4 fictional clients:

- `fjordmat`
- `spareklar`
- `nordvik`
- `skytjenester`

Document types include:

- brand guidelines
- campaign briefs
- campaign reports
- meeting notes
- strategy documents
- customer case material

## Tech Stack

- Python 3.11+
- LangChain
- ChromaDB
- `sentence-transformers`
- `paraphrase-multilingual-MiniLM-L12-v2`
- `uv` for dependency management

Dependencies are defined in `pyproject.toml`.

## How The Pipeline Works

### 1. Load Documents

`src/data_loader.py` loads Markdown files from `data/` and attaches document metadata:

- `client`
- `filename`

### 2. Chunk Documents

`src/chunking.py` performs custom chunking with a few practical improvements:

- paragraph-aware grouping
- sentence-like splitting for oversized sections
- lightweight Markdown-aware handling
- chunk text cleanup before embedding
- stable chunk identity via `chunk_id`

Relevant defaults from `src/config.py`:

- `chunk_size = 1000`
- `chunk_overlap = 200`
- `min_chunk_length = 30`

### 3. Build Embeddings

`src/embeddings.py` initializes a cached embedding client. Current embedding-related improvements include:

- cached client creation with `lru_cache`
- `batch_size` support
- basic parameter validation
- normalized config strings

Current embedding config:

- model: `paraphrase-multilingual-MiniLM-L12-v2`
- device: `cpu`
- normalized embeddings: `True`
- batch size: `32`

### 4. Rebuild The Vector Store

`src/indexing.py` orchestrates the indexing flow.

`src/vectorstore.py` now uses a **rebuild-style** approach:

- existing persisted vector data is removed
- the Chroma store is recreated from the current document set

This is intentional. The current project is a full rebuild pipeline, not an incremental sync pipeline.

### 5. Query The Store

`src/query.py` retrieves relevant chunks from the vector store and includes guardrails for:

- empty queries
- invalid `k`
- invalid `max_query_length`
- missing or empty vector store directory
- query normalization and character-length truncation

### 6. Build Prompt-Ready Context

`src/rag_pipeline.py` currently acts as the Phase 2 demo entrypoint and includes a lightweight pre-LLM orchestration layer:

- `format_source_label(...)`
- `format_source_list(...)`
- `format_retrieved_context(...)`
- `build_prompt(...)`

This means the pipeline already produces:

- retrieved sources
- a prompt-ready context block
- a final prompt string ready to send to an LLM

It does **not** call an LLM yet.

## Stable Chunk Identity

The project now uses one consistent chunk identity through the pipeline:

1. `src/chunking.py` defines `chunk_id`
2. `src/vectorstore.py` stores vectors using that same `chunk_id`
3. `src/rag_pipeline.py` displays the same `chunk_id` in source labels

Example source label:

```text
[Source 1] fjordmat/merkevareretningslinjer.md#chunk-1
```

## Run The Project

Install dependencies with `uv`:

```bash
uv sync
```

Run the demo pipeline:

```bash
uv run python -m src.rag_pipeline
```

Run the Streamlit assistant:

```bash
uv run streamlit run app.py
```

## LLM Configuration

LLM settings are loaded from a local `.env` file. Do not commit real API keys. Use `.env.example` as the template.

The Streamlit sidebar shows:

- `Retrieval only`
- configured LLM providers that have API keys in `.env`

If no provider key is configured, the app still works in retrieval-only mode and shows retrieved sources, context, and prompt trace.

Generic provider configuration:

```bash
LLM_LABEL=Custom OpenAI-compatible
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini
```

OpenAI shortcut:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Anthropic shortcut:

```bash
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

The provider value describes the API protocol, not the company brand. For example, a MiniMax or enterprise gateway that follows OpenAI Chat Completions should use `LLM_PROVIDER=openai_compatible` with its own `LLM_BASE_URL`.

Use `uv run ...` so the declared dependencies are available. Do not assume plain `python3 -m src.rag_pipeline` has the right environment unless you installed dependencies into that interpreter separately.

## What The Demo Currently Prints

Running `src.rag_pipeline` currently:

1. rebuilds the vector store from the local documents
2. runs a small set of demo retrieval questions
3. prints:
   - the question
   - the retrieved source labels
   - the prompt-ready context block
   - the final constructed prompt

## Verified Phase 2 Behavior

The current pipeline has been manually tested end-to-end.

Observed results from the current implementation:

- `Loaded 16 documents`
- `Created 88 chunks`
- `Stored 88 vectors`

This confirms the vector store is rebuilt cleanly rather than accumulating old vectors.

Representative retrieval results:

- `Hva er Fjordmats tone of voice?`
  - top hit: `fjordmat/merkevareretningslinjer.md`
- `Hva var ROAS for Spareklars Google Ads i Q4 2024?`
  - top hit: `spareklar/kampanjerapport-google-ads-q4-2024.md`
- `Hvilke influencere samarbeider Nordvik med?`
  - top hit: `nordvik/influencer-strategi.md`
- `Hvor mange ansatte har LogistikkPartner?`
  - top hit: `skytjenester/kundecase-logistikkpartner.md`

Query guardrails also fail early with clear errors for:

- empty query text
- `k < 1`
- missing vector store directory

## Limitations Right Now

- LLM generation requires `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`/configured model values
- without LLM configuration, the app intentionally falls back to retrieval-only mode
- no hybrid local + web retrieval
- no incremental indexing
- retrieval quality is solid for the demo, but still has room for ranking improvements

## Suggested Next Steps

The most natural next steps from the current codebase are:

1. add concrete `.env` examples for OpenAI-compatible and Anthropic providers
2. test the Streamlit app with a real LLM provider
3. improve the retrieval trace display if needed after UI review
4. later upgrade retrieval with query planning, reranking, and evidence checks

## Notes

- Documents are in Norwegian.
- Code and comments are in English.
