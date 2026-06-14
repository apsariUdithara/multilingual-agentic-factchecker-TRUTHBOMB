import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Local Model Configuration
REASONING_MODEL_PATH = "pipeline-setup/models/phi3-traffic-lora"
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
    'HiruNews',
    'PMDNews',
    'DGINews',
    'news_lk'
]
