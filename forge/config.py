"""Central configuration for NeuralForge."""
import os

# NIM (LLM via TensorRT-LLM)
NIM_URL = os.environ.get("NIM_URL", "http://nim-llm:8000")
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-8b-instruct")

# Triton (embedding + reranking)
TRITON_URL = os.environ.get("TRITON_URL", "http://triton-embed:8000")
TRITON_GRPC_URL = os.environ.get("TRITON_GRPC_URL", "triton-embed:8001")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embedding")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "reranker")
EMBED_CACHE_TTL = int(os.environ.get("EMBED_CACHE_TTL", "300"))
EMBED_CACHE_MAX = int(os.environ.get("EMBED_CACHE_MAX", "500"))
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))

# Qdrant
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "neuralforge")

# cuGraph
GRAPH_DATA_DIR = os.environ.get("GRAPH_DATA_DIR", "data/graph")
GRAPH_RELOAD_INTERVAL = int(os.environ.get("GRAPH_RELOAD_INTERVAL", "300"))

# NeMo Guardrails
GUARDRAILS_CONFIG_DIR = os.environ.get("GUARDRAILS_CONFIG_DIR", "forge/guardrails/config")
GUARDRAILS_ENABLED = os.environ.get("GUARDRAILS_ENABLED", "true").lower() == "true"

# Data
DATA_DIR = os.environ.get("DATA_DIR", "data")
LOG_DIR = os.environ.get("LOG_DIR", "data/logs")

# Server
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8090"))

# Search
SEARCH_DEFAULT_LIMIT = 10
SEARCH_CANDIDATE_MULTIPLIER = 3
BM25_MAX_KEYWORDS = 5
BM25_MIN_KEYWORD_LEN = 4

# Ingestion
SCRAPE_REQUEST_DELAY = float(os.environ.get("SCRAPE_REQUEST_DELAY", "1.5"))
SCRAPE_REQUEST_TIMEOUT = int(os.environ.get("SCRAPE_REQUEST_TIMEOUT", "15"))

# Discovery
DISCOVERY_INTERVAL_HOURS = int(os.environ.get("DISCOVERY_INTERVAL_HOURS", "6"))
DISCOVERY_PAIRS_PER_RUN = int(os.environ.get("DISCOVERY_PAIRS_PER_RUN", "20"))
DISCOVERY_CONFIDENCE_FLOOR = float(os.environ.get("DISCOVERY_CONFIDENCE_FLOOR", "0.6"))

# File Types
SUPPORTED_FILE_TYPES = {
    ".mp3", ".mp4", ".mkv", ".m4a", ".wav", ".flac", ".webm",
    ".txt", ".md", ".pdf", ".json", ".docx", ".html", ".csv",
}
