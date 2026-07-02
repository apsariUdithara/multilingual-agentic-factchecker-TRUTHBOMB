import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Local Model Configuration
# GGUF model is used for fast inference via llama.cpp (claim extraction / local verify)
GGUF_MODEL_PATH = "models/phi3-extractor-unsloth.Q4_K_M.gguf"

# LoRA adapter is used for Transformers-based inference (local researcher tool)
LORA_ADAPTER_PATH = "models/phi3-extractor-lora"

# Legacy compatibility
REASONING_MODEL_PATH = GGUF_MODEL_PATH
TRAFFIC_MODEL_PATH = LORA_ADAPTER_PATH

VECTOR_DB_DIR = "pipeline-setup/vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHROMA_COLLECTION_NAME = "social_media_osint"

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
    "government24.gov.lk",
]

TARGET_CHANNELS = [
    "AdaDeranaNews",
    "newsfirstsl",
    "hirunewsofficial",
    # 'pmdnewssrilanka', # Resolution failing
    # 'news_lk'          # Resolution failing
]

# --- RETRIEVAL & RE-RANKING (Phase 3) ---
# Breadth of each retriever, and the final re-ranked count returned to the graph.
RETRIEVAL_K_DENSE = 5
RETRIEVAL_K_LEXICAL = 5
RETRIEVAL_K_FINAL = 4

# Reciprocal Rank Fusion constant (standard value used across the IR literature).
RRF_K = 60

# Source-tier weights, applied multiplicatively during re-ranking. A document is
# "trusted" if its `source` metadata contains any identifier listed here.
TRUSTED_SOURCE_IDENTIFIERS = TARGET_CHANNELS + TRUSTED_NEWS_DOMAINS
TIER_WEIGHT_TRUSTED = 1.0
TIER_WEIGHT_UNTRUSTED = 0.6

# Recency weights, applied multiplicatively: (max_age_days, weight); first match wins.
RECENCY_WEIGHTS = [(7, 1.0), (30, 0.85), (90, 0.7), (365, 0.5)]
RECENCY_WEIGHT_STALE = 0.35     # older than the last bucket
RECENCY_WEIGHT_UNKNOWN = 0.7    # missing or unparseable timestamp (kept neutral)