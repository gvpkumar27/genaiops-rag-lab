# GenAIOps RAG Lab

GenAIOps RAG Lab is a local-first Retrieval-Augmented Generation platform for secure document question answering, evaluation, and operational observability.

## System Overview

- API service: FastAPI application for `/health`, `/chat`, and `/metrics`
- Retrieval stack: Qdrant vector search with lexical fallback and optional reranking
- Inference stack: Ollama for embedding and response generation
- User interface: Streamlit chat client
- Observability: Prometheus metrics with Grafana dashboards
- Evaluation: Offline benchmark runner and dataset split checks

## Repository Structure

- `app/`: application runtime (API, retrieval, generation, metrics, logging)
- `ui/`: Streamlit interface
- `eval/`: evaluation datasets and runners
- `docs/`: design and operational documentation
- `data/docs/`: local source documents (gitignored)

## Runtime Architecture

1. Client submits a question to `/chat`.
2. API retrieves candidate context from Qdrant.
3. Optional reranker reorders candidates.
4. LLM generates grounded response with inline citations.
5. Guardrails and fallback logic enforce unknown-response behavior when context is insufficient.
6. Metrics and structured events are emitted for observability.

## Structured Logging

Structured logging is enabled.

- Implementation: `app/ops/logging.py`
- Format: JSON Lines (`events.jsonl`)
- Output location: `data/runs/events.jsonl`
- Event fields include: event type, request metadata, timing, and generated `event_id`

## Configuration

Key environment variables:

- `API_HOST` (default: `127.0.0.1`)
- `API_PORT` (default: `8000`)
- `API_KEY` (optional; required for non-localhost exposure)
- `REQUIRE_API_KEY_ON_NON_LOCALHOST` (default: `true`)
- `PUBLIC_RESPONSE_SANITIZE` (default: `true`)
- `QDRANT_URL` (default: `http://localhost:6333`)
- `QDRANT_COLLECTION` (default: `localdocchat`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `CHAT_MODEL` (default: `mistral:7b-instruct`)
- `EMBED_MODEL` (default: `nomic-embed-text`)

## Deployment Prerequisites

- Python 3.9+
- Qdrant available at configured `QDRANT_URL`
- Ollama available at configured `OLLAMA_BASE_URL`
- Docker Desktop (for Prometheus/Grafana stack)

## Standard Operations

### Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Security note:
- `.env` and `.env.observability` are local-only and must never be committed.
- `internal/guardrails_rules.example.json` is a template only; keep real guardrail signatures in local `internal/guardrails_rules.json` (gitignored).

### Ingest Documents

```powershell
cd genaiops-rag-lab
..\.venv\Scripts\python -m app.rag.ingest
```

### Run API

```powershell
cd genaiops-rag-lab
..\.venv\Scripts\python -m app
```

### Run UI

```powershell
cd ..
.\.venv\Scripts\streamlit run .\genaiops-rag-lab\ui\streamlit_app.py
```

### Run Observability Stack

```powershell
cd genaiops-rag-lab
Copy-Item .env.observability.example .env.observability
docker compose --env-file .env.observability -f docker.compose.observability.yml up -d
```

Endpoints:

- API health: `http://localhost:8000/health`
- API metrics: `http://localhost:8000/metrics`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Streamlit UI: `http://localhost:8501`

## Security Posture

- Local-safe defaults (`127.0.0.1` bindings where applicable)
- API key enforcement for non-localhost deployment
- Public response sanitization for citation metadata
- No private documents or secrets committed to repository (`data/docs/`, `.env*` protected by gitignore)

## Evaluation

Run baseline evaluation:

```powershell
cd genaiops-rag-lab
..\.venv\Scripts\python .\eval\run_eval.py
```

## Test Command

```powershell
.\.venv\Scripts\python -m pytest -q
```
