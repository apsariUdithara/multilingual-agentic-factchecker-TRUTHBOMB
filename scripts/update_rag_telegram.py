"""One-shot Telegram -> ChromaDB ingestion for the OSINT knowledge base.

Credentials are read from ``.env`` via ``src.config`` (previously they were
hard-coded here). Messages are de-duplicated by id, tagged with metadata, and
chunked before indexing into the canonical collection.

Run from the project root:  ``python -m scripts.update_rag_telegram``
"""
import asyncio
import hashlib
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telethon import TelegramClient
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import (
    VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME,
    TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_SESSION_NAME, TARGET_CHANNELS,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MESSAGES_PER_CHANNEL = 50


async def main() -> None:
    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        logger.error("TELEGRAM_API_ID / TELEGRAM_API_HASH not set in .env. Aborting.")
        return

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )

    # Existing message ids, so re-runs don't create duplicates.
    existing_ids = set()
    try:
        current = vector_db.get(include=["metadatas"])
        existing_ids = {
            m.get("message_id") for m in current.get("metadatas", []) if m and m.get("message_id")
        }
    except Exception:
        pass

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    client = TelegramClient(TELEGRAM_SESSION_NAME, TELEGRAM_API_ID, TELEGRAM_API_HASH)
    await client.start()

    new_docs, seen_hashes = [], set()
    for channel in TARGET_CHANNELS:
        async for message in client.iter_messages(channel, limit=MESSAGES_PER_CHANNEL):
            if not message.text:
                continue
            msg_id = str(message.id)
            if msg_id in existing_ids:
                continue
            content_hash = hashlib.md5(message.text.strip().lower().encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            existing_ids.add(msg_id)

            meta = {
                "source": f"Telegram-@{channel}",
                "message_id": msg_id,
                "timestamp": message.date.isoformat() if message.date
                else datetime.now(timezone.utc).isoformat(),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "kind": "osint_alert",
            }
            new_docs.append(Document(page_content=message.text, metadata=meta))

    await client.disconnect()

    if not new_docs:
        logger.info("No new messages to ingest.")
        return

    chunks = splitter.split_documents(new_docs)
    vector_db.add_documents(chunks)
    logger.info("Ingested %d new messages (%d chunks) into '%s'.",
                len(new_docs), len(chunks), CHROMA_COLLECTION_NAME)


if __name__ == "__main__":
    asyncio.run(main())
