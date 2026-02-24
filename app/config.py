import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_KEY = os.getenv("API_KEY", "").strip()
    REQUIRE_API_KEY_ON_NON_LOCALHOST = os.getenv("REQUIRE_API_KEY_ON_NON_LOCALHOST", "true").lower() == "true"
    PUBLIC_RESPONSE_SANITIZE = os.getenv("PUBLIC_RESPONSE_SANITIZE", "true").lower() == "true"

    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "localdocchat")

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral:7b-instruct")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

    TOP_K = int(os.getenv("TOP_K", "6"))
    RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "15"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

    ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "1"))
    OLLAMA_SEED = int(os.getenv("OLLAMA_SEED", "7"))
    STABLE_QUERY_NORMALIZATION = os.getenv("STABLE_QUERY_NORMALIZATION", "true").lower() == "true"
    FAITHFULNESS_MIN_GROUNDED = float(os.getenv("FAITHFULNESS_MIN_GROUNDED", "0.30"))
    ENABLE_CONTAMINATION_FILTER = os.getenv("ENABLE_CONTAMINATION_FILTER", "true").lower() == "true"

settings = Settings()
