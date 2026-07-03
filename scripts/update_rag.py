"""Initialize and reset the TruthBomb vector store.

Clears stale collections and prepares the database for live OSINT/Telegram ingestion.
Run from the project root:  ``python -m scripts.update_rag``
"""
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (
    VECTOR_DB_DIR, CHROMA_COLLECTION_NAME, STALE_COLLECTIONS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def reset_and_initialize_db() -> None:
    """Clear stale collections and reset the canonical collection so it is ready for live data."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        existing = {c.name for c in client.list_collections()}
        
        # Drop historical stale collections
        for name in STALE_COLLECTIONS:
            if name in existing:
                client.delete_collection(name)
                logger.info("Dropped stale collection: %s", name)
                
        # Reset the canonical collection to keep the database fresh and verified
        if CHROMA_COLLECTION_NAME in existing:
            client.delete_collection(CHROMA_COLLECTION_NAME)
            logger.info("Reset canonical collection: %s", CHROMA_COLLECTION_NAME)
            
        logger.info("RAG Vector Database successfully initialized. Ready for live ingestion.")
    except Exception as exc:
        logger.error("Could not initialize database: %s", exc)


if __name__ == "__main__":
    reset_and_initialize_db()
