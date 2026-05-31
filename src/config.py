import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Local Model Configuration
TRAFFIC_MODEL_PATH = "pipeline-setup/models/phi3-traffic-lora"
VECTOR_DB_DIR = "pipeline-setup/vector_db"
