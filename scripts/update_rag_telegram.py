import os
import time
import logging
from datetime import datetime
from telethon.sync import TelegramClient
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# =====================================================================
# SYSTEM CONFIGURATION
# =====================================================================
VECTOR_DB_DIR = "D:/7th/AI/Project/chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "traffic_telemetry"

# --- TELEGRAM CREDENTIALS ---
API_ID = 38020229  # Replace with your API ID (Integer)
API_HASH = 'bad58fa93d224c8320db9639efa7ed5e'  # Replace with your API Hash (String)
SESSION_NAME = os.path.join(os.path.dirname(__file__), 'rag_updater_session')

# Target Channels (Public usernames)
TARGET_CHANNELS = ['AdaDeranaNews'] 

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class TelegramRAGUpdater:
    def __init__(self):
        logger.info("Initializing Telegram MTProto pipeline...")
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
        self.client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

    def _get_existing_ids(self) -> set:
        try:
            current_data = self.vector_db.get(include=["metadatas"])
            if current_data and "metadatas" in current_data:
                return {
                    meta.get("message_id") 
                    for meta in current_data["metadatas"] 
                    if meta and "message_id" in meta
                }
        except Exception as e:
            logger.warning(f"Could not read metadata cache: {e}")
        return set()

    def fetch_and_update(self):
        existing_ids = self._get_existing_ids()
        new_documents = []

        with self.client:
            for channel in TARGET_CHANNELS:
                logger.info(f"Querying Telegram Channel: @{channel}")
                # Fetch the 20 most recent messages
                for message in self.client.iter_messages(channel, limit=20):
                    if not message.text:
                        continue
                        
                    msg_id = str(message.id)
                    if msg_id in existing_ids:
                        continue

                    # Create normalized schema
                    doc = Document(
                        page_content=f"Source: {channel}\nAlert: {message.text}",
                        metadata={
                            "source": f"Telegram-@{channel}",
                            "message_id": msg_id,
                            "timestamp": message.date.isoformat(),
                            "ingested_at": datetime.utcnow().isoformat()
                        }
                    )
                    new_documents.append(doc)
                    existing_ids.add(msg_id)

        if new_documents:
            logger.info(f"Injecting {len(new_documents)} new Telegram alerts into ChromaDB...")
            self.vector_db.add_documents(new_documents)
            logger.info("Database successfully updated.")
        else:
            logger.info("Sync complete. No new messages detected.")

if __name__ == "__main__":
    updater = TelegramRAGUpdater()
    # The first time you run this, it will prompt for your phone number in the console 
    # to authenticate the session. It saves a .session file so you only do it once.
    updater.fetch_and_update()