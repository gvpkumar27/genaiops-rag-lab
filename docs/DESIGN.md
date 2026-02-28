# GenAIOps RAG Lab Public Design

## 1. Purpose
GenAIOps RAG Lab is a local-first document Q&A system that retrieves context from
indexed documents and returns grounded answers with citations.

## 2. Scope
In scope:
- Local document ingestion (`PDF`, `TXT`, `MD`)
- Chunking, embedding, and vector indexing
- Retrieval and optional reranking
- Grounded answer generation
- Streamlit UI + FastAPI backend
- Prometheus/Grafana observability

Out of scope:
- Multi-tenant production platform features
- Distributed orchestration/scaling
- Model fine-tuning workflows

## 3. Architecture
Main components:
1. UI: `ui/streamlit_app.py`
2. API: `app/__main__.py`
3. Ingest: `app/rag/ingest.py`
4. Retrieval: `app/rag/retrieve.py`
5. Rerank (optional): `app/rag/rerank.py`
6. Generation: `app/rag/generate.py`, `app/rag/prompts.py`
7. Vector DB: Qdrant
8. Model runtime: Ollama

## 4. Data Flow
### Ingestion
1. Read files from `data/docs`
2. Normalize and chunk text
3. Embed chunks
4. Upsert vectors + payload into Qdrant

### Query
1. UI sends question to `POST /chat`
2. API validates request and policies
3. Retrieval finds relevant context
4. Optional rerank refines ordering
5. Generation produces grounded answer
6. Response includes citations + timing metadata

## 5. API Surface
- `GET /health`
- `GET /metrics`
- `POST /chat`

## 6. Safety and Reliability (Public Summary)
- Grounded-response behavior with citation-focused output
- Guardrails for harmful or disallowed prompt categories
- Out-of-scope blocking for low-coverage queries
- Fallback handling for low-confidence outputs
- Dependency failure paths return explicit service errors

## 7. Configuration (High-Level)
Configuration is environment-driven (`app/config.py`), including:
- API host/port/security settings
- Retrieval/chunking settings
- Model/runtime settings
- Guardrails and observability toggles

Use `.env.example` and `.env.observability.example` as templates only.

## 8. Observability
The service exposes Prometheus metrics and supports Grafana dashboards for:
- Request volume and failures
- End-to-end and stage latencies
- Cache/fallback/blocking behavior
- Index size and ingest counters

## 9. Security Notes
- Keep `.env` files local-only (never commit secrets)
- Keep private rule/signature files in `internal/` (gitignored)
- Public docs and examples are intentionally sanitized

## 10. File Map
- `app/__main__.py`: request orchestration
- `app/config.py`: runtime config
- `app/schemas.py`: API schemas
- `app/rag/*`: RAG pipeline modules
- `app/ops/*`: metrics/logging/http helpers
- `ui/streamlit_app.py`: user interface
- `docker.compose.yml`: local Qdrant
- `docker.compose.observability.yml`: Prometheus/Grafana stack
