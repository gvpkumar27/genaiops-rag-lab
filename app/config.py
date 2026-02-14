import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "localdocchat")

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral:7b-instruct")
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

    TOP_K = int(os.getenv("TOP_K", "5"))
    RETRIEVE_CANDIDATES = int(os.getenv("RETRIEVE_CANDIDATES", "20"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

    ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v1")

settings = Settings()
