# GenAIOps RAG Lab

Local document Q&A app using FastAPI + Qdrant + Ollama + Streamlit.

## Project Layout

- `genaiops-rag-lab/app`: API and RAG pipeline code
- `genaiops-rag-lab/ui/streamlit_app.py`: Streamlit UI
- `genaiops-rag-lab/app/rag/ingest.py`: document ingestion script
- `genaiops-rag-lab/eval/run_eval.py`: simple evaluation runner
- `genaiops-rag-lab/data/docs`: place your source documents here

## Prerequisites

1. Python 3.9+
2. Qdrant running on `http://localhost:6333`
3. Ollama running on `http://localhost:11434`
4. Ollama models pulled (defaults):
   - `mistral:7b-instruct`
   - `nomic-embed-text`

## Setup

From repo root (`genaiops-rag-lab_project`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r dev-requirements.txt
```

Alternative (single command for app + dev extras):

```powershell
pip install -e ".[dev]"
```

## Configure (optional)

Set environment variables if needed:

- `API_HOST` (default: `127.0.0.1`)
- `API_PORT` (default: `8000`)
- `API_KEY` (default: empty; when set, `/chat` and `/metrics` require `x-api-key`)
- `REQUIRE_API_KEY_ON_NON_LOCALHOST` (default: `true`; refuses non-localhost bind without API key)
- `PUBLIC_RESPONSE_SANITIZE` (default: `true`; sanitizes citation metadata in API responses for public safety)
- `QDRANT_URL` (default: `http://localhost:6333`)
- `QDRANT_COLLECTION` (default: `localdocchat`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `CHAT_MODEL` (default: `mistral:7b-instruct`)
- `EMBED_MODEL` (default: `nomic-embed-text`)
- `TOP_K` (default: `6`)
- `RETRIEVE_CANDIDATES` (default: `15`)
- `CHUNK_SIZE` (default: `600`)
- `CHUNK_OVERLAP` (default: `100`)
- `ENABLE_RERANK` (default: `true`)
- `ENABLE_CACHE` (default: `true`)
- `OLLAMA_TEMPERATURE` (default: `0`)
- `OLLAMA_TOP_P` (default: `1`)
- `OLLAMA_SEED` (default: `7`)
- `STABLE_QUERY_NORMALIZATION` (default: `true`)
- `FAITHFULNESS_MIN_GROUNDED` (default: `0.30`)
- `ENABLE_CONTAMINATION_FILTER` (default: `true`)

## Ingest Documents

1. Put files under `genaiops-rag-lab/data/docs` (`.pdf`, `.txt`, `.md`, etc.).
2. Run ingest from `genaiops-rag-lab` folder:

```powershell
cd genaiops-rag-lab
..\.venv\Scripts\python -m app.rag.ingest
```

Note: `data/docs/` is gitignored, so your local source files are not committed to the repo.

## Run API

From `genaiops-rag-lab` folder:

```powershell
..\.venv\Scripts\python -m app
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Metrics endpoint (Prometheus format):

```powershell
Invoke-WebRequest http://localhost:8000/metrics | Select-Object -ExpandProperty Content
```

If `API_KEY` is set:

```powershell
Invoke-WebRequest http://localhost:8000/metrics -Headers @{"x-api-key"="<your-api-key>"} | Select-Object -ExpandProperty Content
```

## Security Notes (Local/Dev Defaults)

- API defaults to `127.0.0.1:8000` (`API_HOST`/`API_PORT` are configurable).
- When `API_KEY` is set, `/chat` and `/metrics` require header `x-api-key`.
- By default, app startup is blocked if `API_HOST` is non-localhost and `API_KEY` is empty.
- If you expose this beyond localhost (`API_HOST=0.0.0.0`), set a strong `API_KEY` and use trusted network controls.

Key metrics exposed:
- `chat_requests_total`
- `chat_errors_total{type=...}`
- `chat_latency_seconds`
- `retrieve_latency_seconds`
- `rerank_latency_seconds`
- `generate_latency_seconds`
- `fallback_answers_total`
- `cache_hits_total`
- `docs_indexed_total`
- `qdrant_collection_points`

## Run Observability Stack (Prometheus + Grafana)

From `genaiops-rag-lab` folder:

```powershell
Copy-Item .env.observability.example .env.observability
# Edit .env.observability and set GRAFANA_ADMIN_PASSWORD
docker compose --env-file .env.observability -f docker.compose.observability.yml up -d
```

Note: Example values in `.env.observability.example` are placeholders; do not use them in production.

Endpoints:
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (login from `.env.observability`)

Grafana auto-loads:
- data source: `Prometheus`
- dashboard: `LocalDocChat Overview`

Notes:
- Keep FastAPI running on `http://localhost:8000` so Prometheus can scrape `/metrics`.
- On Docker Desktop (Windows/Mac), Prometheus uses `host.docker.internal:8000`.

## Run UI

From repo root:

```powershell
.\.venv\Scripts\streamlit run .\genaiops-rag-lab\ui\streamlit_app.py
```

UI security options:
- `PUBLIC_UI_MODE` (default: `true`) hides raw source path/chunk/score details in citations.
- `SHOW_CITATION_DEBUG` (default: `false`) re-enables raw citation debug details for internal use.

Note:
- With `PUBLIC_RESPONSE_SANITIZE=true`, `/chat` returns sanitized citations (`Document N`) and removes raw chunk/score/preview details.
- Set `PUBLIC_RESPONSE_SANITIZE=false` only for trusted internal debugging.
- Keep API access authenticated for any public exposure.

## Run Eval

From `genaiops-rag-lab` folder:

```powershell
..\.venv\Scripts\python .\eval\run_eval.py
```

`eval/golden_qna.jsonl` format is JSON Lines (one JSON object per line):

```jsonl
{"question":"What is Retrieval Augmented Generation (RAG)?","expected_keywords":["retrieval","context","documents","generate"]}
```

Fields:
- `question`: string
- `expected_keywords`: list of strings used by simple keyword-match scoring in `eval/run_eval.py`

Tip: use short, forgiving keywords (3-5 per question) because scoring is substring-based.

Public repo best practice:
- Keep `eval/*` public files sanitized/synthetic.
- Put internal golden references (full answer keys/citation targets) under `eval/private/` (gitignored).

## Docker Notes

- `docker.compose.yml` maps Qdrant as `127.0.0.1:6333:6333` for local-only access by default.

## Run Tests

From repo root:

```powershell
.\.venv\Scripts\python -m pytest -q
```

