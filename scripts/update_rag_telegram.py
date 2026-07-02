"""
Standalone Telegram ingestion worker for the TruthBomb knowledge base.

Polls the configured Telegram news channels, deduplicates against messages already
stored in ChromaDB (by ``message_id``), embeds new posts, and writes them to the
vector store. This runs as its own process, fully decoupled from the FastAPI
request path (Phase 1). This corresponds to the "News Sources -> Ingest Worker ->
Embed and Store" branch of the system architecture.

Usage:
    python -m scripts.update_rag_telegram            # poll continuously
    python -m scripts.update_rag_telegram --once     # single pass, then exit

Required environment variables (.env):
    TELEGRAM_API_ID        numeric app id from https://my.telegram.org
    TELEGRAM_API_HASH      app hash string
    TELEGRAM_SESSION       (optional) session name, default 'rag_updater_session'
    INGEST_POLL_SECONDS    (optional) poll interval in seconds, default 60
    INGEST_CHANNEL_LIMIT   (optional) messages fetched per channel per pass, default 20

Note: the first run triggers an interactive Telegram login to create the local
session file under scripts/. Subsequent runs reuse that session.
"""

import os
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import Set, Tuple

from dotenv import load_dotenv
import hashlib
import logging
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telethon import TelegramClient
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL_NAME,
    CHROMA_COLLECTION_NAME,
    TARGET_CHANNELS,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest_worker")

SESSION_NAME = os.path.join("scripts", os.getenv("TELEGRAM_SESSION", "rag_updater_session"))
POLL_SECONDS = int(os.getenv("INGEST_POLL_SECONDS", "60"))
CHANNEL_LIMIT = int(os.getenv("INGEST_CHANNEL_LIMIT", "20"))


def _load_credentials() -> Tuple[int, str]:
    """Read Telegram API credentials from the environment (never hardcoded)."""
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        raise RuntimeError(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in the environment (.env)."
        )
    return int(api_id), api_hash


def _get_vector_db() -> Chroma:
    """Open the persisted ChromaDB collection with the project's embedding model."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )


def _existing_message_ids(vector_db: Chroma) -> Set[str]:
    """Return the set of message_ids already stored, used to skip duplicates."""
    try:
        current = vector_db.get(include=["metadatas"])
    except Exception as exc:  # noqa: BLE001 - dedup is best-effort; never fatal
        logger.warning("Could not read existing metadata for dedup: %s", exc)
        return set()
    metadatas = (current or {}).get("metadatas") or []
    return {m.get("message_id") for m in metadatas if m and m.get("message_id")}


async def _poll_once(client: TelegramClient, vector_db: Chroma) -> int:
    """Fetch new messages from all target channels and store them. Returns count added."""
    existing_ids = _existing_message_ids(vector_db)
    new_documents = []

    for channel in TARGET_CHANNELS:
        try:
            async for message in client.iter_messages(channel, limit=CHANNEL_LIMIT):
                if not message.text:
                    continue
                msg_id = str(message.id)
                if msg_id in existing_ids:
                    continue
                new_documents.append(
                    Document(
                        page_content=f"Source: {channel}\nAlert: {message.text}",
                        metadata={
                            "source": f"Telegram-@{channel}",
                            "message_id": msg_id,
                            "timestamp": message.date.isoformat()
                            if message.date
                            else datetime.now(timezone.utc).isoformat(),
                            "ingested_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                )
                existing_ids.add(msg_id)
        except Exception as exc:  # noqa: BLE001 - one bad channel must not abort the pass
            logger.error("Failed polling channel %s: %s", channel, exc)

    if new_documents:
        vector_db.add_documents(new_documents)
        logger.info("Stored %d new messages in ChromaDB.", len(new_documents))
    else:
        logger.info("No new messages this pass.")
    return len(new_documents)


async def run(poll_forever: bool) -> None:
    """Start the Telegram client and poll once or continuously."""
    api_id, api_hash = _load_credentials()
    vector_db = _get_vector_db()

    client = TelegramClient(SESSION_NAME, api_id, api_hash)
    await client.start()  # interactive login on first run; reuses session afterwards
    logger.info("Telegram client started. Channels: %s", ", ".join(TARGET_CHANNELS))

    try:
        while True:
            await _poll_once(client, vector_db)
            if not poll_forever:
                break
            await asyncio.sleep(POLL_SECONDS)
    finally:
        await client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="TruthBomb Telegram ingestion worker.")
    parser.add_argument("--once", action="store_true", help="Run a single pass then exit.")
    args = parser.parse_args()
    asyncio.run(run(poll_forever=not args.once))


if __name__ == "__main__":
    main()