import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Local Model Configuration
# GGUF model is used for fast inference via llama.cpp (used in main.py)
GGUF_MODEL_PATH = "models/phi3-extractor-unsloth.Q4_K_M.gguf"

# LoRA adapter is used for Transformers-based inference (used in agents)
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
    "government24.gov.lk"
]

TARGET_CHANNELS = [
    'AdaDeranaNews',
    'newsfirstsl',
    'hirunewsofficial',
    # 'pmdnewssrilanka', # Resolution failing
    # 'news_lk'          # Resolution failing
]
