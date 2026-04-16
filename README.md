# RAG Knowledge Assistant

This repository is a demo RAG project. The idea is an internal client knowledge assistant for a marketing agency: agency employees can search client material such as brand guidelines, campaign briefs, reports, and strategy documents and retrieve relevant source-backed context.

The project currently delivers a solid Phase 2 retrieval pipeline and a pre-LLM orchestration layer. It can:

- load fictional client documents from `data/`
- split them into retrieval chunks
- generate embeddings locally
- rebuild a Chroma vector store from a clean state
- run retrieval queries against that store
- format retrieved chunks into prompt-ready context with stable source labels

It does **not** yet include a Streamlit UI or final answer generation from an LLM.

## Current Status

Implemented now:

- modular RAG pipeline under `src/`
- rebuild-style indexing pipeline
- stable `chunk_id` identity from chunking to storage to display
- query input guardrails
- prompt-ready context orchestration in `src.rag_pipeline`

Not implemented yet:

- `app.py` / Streamlit UI
- LLM answer generation
- hybrid local + web retrieval

## Project Structure

```text
RAG-knowledge-assistant/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── data/
│   ├── fjordmat/
│   ├── nordvik/
│   ├── skytjenester/
│   └── spareklar/
├── src/
│   ├── __init__.py
│   ├── chunking.py
│   ├── config.py
│   ├── data_loader.py
│   ├── embeddings.py
│   ├── indexing.py
│   ├── query.py
│   ├── rag_pipeline.py
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

- no final LLM answer generation
- no UI
- no client filter in a frontend
- no hybrid local + web retrieval
- no incremental indexing
- retrieval quality is solid for the demo, but still has room for ranking improvements

## Suggested Next Steps

The most natural next steps from the current codebase are:

1. connect an LLM to the already-built prompt construction flow
2. add `app.py` with a Streamlit chat interface
3. show answer text together with source citations
4. optionally add client filtering in the UI
5. optionally explore hybrid retrieval later

## Notes

- Documents are in Norwegian.
- Code and comments are in English.
