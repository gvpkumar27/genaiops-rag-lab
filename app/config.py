"""Runtime configuration loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


def _is_true(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


class Settings:
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_KEY = os.getenv("API_KEY", "").strip()
    REQUIRE_API_KEY_ON_NON_LOCALHOST = _is_true("REQUIRE_API_KEY_ON_NON_LOCALHOST")
    PUBLIC_RESPONSE_SANITIZE = _is_true("PUBLIC_RESPONSE_SANITIZE")

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "localdocchat")

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral:7b-instruct")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

    TOP_K = int(os.getenv("TOP_K", "6"))
    RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "15"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    ENABLE_RERANK = _is_true("ENABLE_RERANK")
    ENABLE_CACHE = _is_true("ENABLE_CACHE")
    PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").strip()
    SYSTEM_PROMPT_FILE = os.getenv("SYSTEM_PROMPT_FILE", "").strip()
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "1"))
    OLLAMA_SEED = int(os.getenv("OLLAMA_SEED", "7"))
    HTTP_RETRIES = int(os.getenv("HTTP_RETRIES", "2"))
    HTTP_RETRY_BACKOFF_SEC = float(os.getenv("HTTP_RETRY_BACKOFF_SEC", "0.5"))
    STABLE_QUERY_NORMALIZATION = _is_true("STABLE_QUERY_NORMALIZATION")
    FAITHFULNESS_MIN_GROUNDED = float(os.getenv("FAITHFULNESS_MIN_GROUNDED", "0.30"))
    ENABLE_CONTAMINATION_FILTER = _is_true("ENABLE_CONTAMINATION_FILTER")
    ENABLE_PROMPT_GUARDRAILS = _is_true("ENABLE_PROMPT_GUARDRAILS")
    ENABLE_OUT_OF_SCOPE_GUARD = _is_true("ENABLE_OUT_OF_SCOPE_GUARD")
    OUT_OF_SCOPE_MIN_COVERAGE = float(os.getenv("OUT_OF_SCOPE_MIN_COVERAGE", "0.18"))
    GUARDRAILS_FILE = os.getenv("GUARDRAILS_FILE", "").strip()
    REQUIRE_PRIVATE_GUARDRAILS = _is_true("REQUIRE_PRIVATE_GUARDRAILS", "false")


settings = Settings()
