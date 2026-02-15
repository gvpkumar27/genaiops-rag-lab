# Design Document: LocalDocChat (RAG)

## 1. Objective
Build a local, document-grounded question-answering system that:
- ingests user documents,
- indexes chunks in a vector store,
- retrieves relevant context per query,
- generates grounded answers via a local LLM,
- serves end users through a simple web UI.

Primary non-functional goals:
- run fully local,
- keep operational setup simple,
- fail safely when dependencies are unavailable,
- answer only from uploaded documents.

## 2. System Scope
In scope:
- PDF/TXT/MD ingestion
- chunking + embedding
- Qdrant vector indexing/search
- retrieval + optional rerank
- answer generation via Ollama
- Streamlit UI for chat

Out of scope:
- multi-user auth/tenancy
- distributed scaling
- enterprise observability stack
- fine-tuned model training

## 3. High-Level Architecture
Components:
1. Streamlit UI (`ui/streamlit_app.py`)
2. FastAPI backend (`app/__main__.py`)
3. Ingestion pipeline (`app/rag/ingest.py`)
4. Retrieval pipeline (`app/rag/retrieve.py`)
5. Generation pipeline (`app/rag/generate.py`)
6. Vector DB: Qdrant
7. Model runtime: Ollama
8. Local data dirs: `data/docs`, `data/cache`, `data/runs`

External services:
- Qdrant at `QDRANT_URL`
- Ollama at `OLLAMA_BASE_URL`

## 4. Data Flow
### 4.1 Ingestion Flow
1. Read files from `data/docs`.
2. Extract text (`pypdf` for PDFs).
3. Normalize noisy OCR/PDF text (including fragmented letter cleanup).
4. Chunk text with overlap (`chunk_size`, `chunk_overlap`).
5. Generate embeddings for each chunk via Ollama embedding model.
6. Recreate target Qdrant collection and upsert chunk vectors + payload.

Payload fields per chunk:
- `source`
- `chunk_id`
- `text`
- `type`

### 4.2 Query Flow
1. UI posts `{ question }` to `POST /chat`.
2. Backend checks cache (if enabled).
3. Embed question.
4. Retrieve candidates from Qdrant.
5. Hybrid ranking:
- semantic vector score
- keyword overlap score
- lexical fallback pass via Qdrant scroll
6. Optional LLM rerank (non-blocking on failure).
7. Generate final answer from selected contexts.
8. If model returns low-quality/"I don't know", apply extractive fallback from retrieved chunks.
9. Return answer + citations + metadata.

## 5. API Design
### 5.1 `GET /health`
Returns service health:
```json
{"status": "ok"}
```

### 5.2 `POST /chat`
Request:
```json
{"question": "...", "filters": null}
```
Response:
- `answer: str`
- `citations: [{source, chunk_id, score, text_preview}]`
- `meta: {model, embed_model, retrieve_sec, generate_sec, ...}`

Error handling:
- `400` invalid input
- `503` dependency unavailable (Qdrant/Ollama)
- `500` unhandled backend exception

## 6. Configuration
Configured via `.env` and `app/config.py`.
Key variables:
- `QDRANT_URL`, `QDRANT_COLLECTION`
- `OLLAMA_BASE_URL`
- `CHAT_MODEL`, `EMBED_MODEL`
- `TOP_K`, `RETRIEVE_CANDIDATES`
- `CHUNK_SIZE`, `CHUNK_OVERLAP`
- `ENABLE_RERANK`, `ENABLE_CACHE`
- `PROMPT_VERSION`

Recommended local profile:
- smaller chat model for responsiveness
- `ENABLE_RERANK=false` unless model latency is acceptable

## 7. Operational Model
## 7.1 Startup
1. Start Qdrant (Docker).
2. Ensure Ollama is running + required models are pulled.
3. Run ingestion.
4. Start FastAPI.
5. Start Streamlit.

## 7.2 Runtime Artifacts
- Cache files: `data/cache`
- Event logs: `data/runs/events.jsonl`
- Source docs: `data/docs`

## 8. Reliability and Failure Strategy
- explicit HTTP errors for dependency failures
- rerank failure does not block answer generation
- generation fallback paths for incompatible Ollama API versions (`/api/chat` -> `/api/generate`)
- extractive fallback when generative output is low quality

## 9. Security Considerations
Current posture (local app):
- no authn/authz
- no rate limiting
- local file access assumptions

Hardening recommendations:
- add auth for non-local deployments
- validate/limit uploaded file types and sizes
- sanitize path/file serving endpoints
- add request timeouts and abuse controls

## 10. Performance Considerations
- ingestion is O(number_of_chunks)
- retrieval latency depends on embedding + Qdrant search
- generation dominates p95 latency for small local models
- cache reduces repeated-query latency

Tuning knobs:
- `TOP_K`, `RETRIEVE_CANDIDATES`
- smaller `CHAT_MODEL`
- disable rerank for faster responses
- chunk size/overlap tradeoff for recall vs context quality

## 11. Testing Strategy
Current:
- unit tests under `tests/`

Recommended additions:
- retrieval relevance regression tests
- ingestion normalization tests (PDF/OCR artifacts)
- end-to-end smoke tests for `/chat`
- failure-path tests (Qdrant down, Ollama down, model missing)

## 12. Known Limitations
- quality depends on PDF extraction quality
- strict document grounding can return "I don't know" for paraphrased/out-of-scope questions
- local model performance varies by hardware
- single-instance architecture only

## 13. Future Enhancements
- richer source links and page-level citations
- upload API + UI document management
- structured retrieval diagnostics endpoint
- improved prompt templates per document type
- optional hybrid BM25 index alongside vectors

## 14. File/Module Map
- `app/__main__.py`: FastAPI entrypoint and chat orchestration
- `app/config.py`: runtime settings
- `app/schemas.py`: request/response schema
- `app/rag/ingest.py`: ingestion and indexing
- `app/rag/retrieve.py`: retrieval and ranking
- `app/rag/rerank.py`: optional LLM rerank
- `app/rag/generate.py`: answer generation
- `ui/streamlit_app.py`: end-user chat UI
- `docker.compose.yml`: local Qdrant service
