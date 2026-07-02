import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Telegram MTProto credentials (never hard-code; read from .env) ---
# Get these from https://my.telegram.org
TELEGRAM_API_ID = int(os.getenv("TELEGRAM_API_ID", "0") or "0")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH", "")
# Session file lives next to the scripts by default; overridable via env.
TELEGRAM_SESSION_NAME = os.getenv(
    "TELEGRAM_SESSION_NAME",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "rag_updater_session"),
)

# Local Model Configuration
# GGUF model is used for fast inference via llama.cpp (used in main.py)
GGUF_MODEL_PATH = "models/phi3-extractor-unsloth.Q4_K_M.gguf"

# LoRA adapter is used for Transformers-based inference (used in agents)
LORA_ADAPTER_PATH = "models/phi3-extractor-lora"

VECTOR_DB_DIR = "pipeline-setup/vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Single source of truth for the collection name. The retriever, ingestion
# scripts and the API all import this so they can never drift apart again.
CHROMA_COLLECTION_NAME = "social_media_osint"
# Any collections left over from earlier projects (e.g. the traffic-law demo)
# are dropped by scripts/update_rag.py so the store stays clean.
STALE_COLLECTIONS = ["traffic_laws"]

# --- Retrieval / RAG tuning ---
# How many candidates each retriever (dense, sparse) contributes before fusion.
RETRIEVAL_CANDIDATES = 10
# Final number of context chunks handed to the reasoning LLM after re-ranking.
RETRIEVAL_TOP_N = 5
# Reciprocal Rank Fusion constant (60 is the value from the original RRF paper).
RRF_K = 60
# Optional multilingual cross-encoder re-ranker. Set ENABLE_RERANKER=0 to skip
# it (e.g. on CPU-only machines); retrieval then falls back to RRF ordering.
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "1") not in ("0", "false", "False")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-v2-m3")

# --- Chunking (used at ingestion time) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- TRUSTED SOURCES ---
TRUSTED_NEWS_DOMAINS = [
    "adaderana.lk",
    "newsfirst.lk",
    "dailymirror.lk",
    "ft.lk",
    "news.lk",
    "lankadeepa.lk",
    "virakesari.lk",
    "economynext.com",
    "island.lk",
    "government24.gov.lk"
]

TARGET_CHANNELS = [
    'AdaDeranaNews',
    'newsfirstsl',
    'hirunewsofficial',
    # 'pmdnewssrilanka', # Resolution failing
    # 'news_lk'          # Resolution failing
]
