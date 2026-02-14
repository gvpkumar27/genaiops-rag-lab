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

## Configure (optional)

Set environment variables if needed:

- `QDRANT_URL` (default: `http://localhost:6333`)
- `QDRANT_COLLECTION` (default: `localdocchat`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `CHAT_MODEL` (default: `mistral:7b-instruct`)
- `EMBED_MODEL` (default: `nomic-embed-text`)
- `TOP_K` (default: `5`)
- `CHUNK_SIZE` (default: `900`)
- `CHUNK_OVERLAP` (default: `150`)
- `ENABLE_RERANK` (default: `true`)
- `ENABLE_CACHE` (default: `true`)

## Ingest Documents

1. Put files under `genaiops-rag-lab/data/docs` (`.pdf`, `.txt`, `.md`, etc.).
2. Run ingest from `genaiops-rag-lab` folder:

```powershell
cd genaiops-rag-lab
..\.venv\Scripts\python -m app.rag.ingest
```

## Run API

From `genaiops-rag-lab` folder:

```powershell
..\.venv\Scripts\python -m app
```

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Run UI

From repo root:

```powershell
.\.venv\Scripts\streamlit run .\genaiops-rag-lab\ui\streamlit_app.py
```

## Run Eval

From `genaiops-rag-lab` folder:

```powershell
..\.venv\Scripts\python .\eval\run_eval.py
```

## Run Tests

From repo root:

```powershell
.\.venv\Scripts\python -m pytest -q
```
