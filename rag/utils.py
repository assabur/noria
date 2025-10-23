from __future__ import annotations
import os
from dotenv import load_dotenv


load_dotenv()


QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8")


EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "intfloat/multilingual-e5-base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
LANG_FILTER = os.getenv("LANG_FILTER", "en")