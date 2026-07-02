"""Populate the TruthBomb vector store from the local OSINT dataset.

This replaces the previous version, which:
  * ingested only the one-line ``core_event`` headline per record (nothing to
    retrieve), and
  * wrote into whatever collection happened to be default, leaving the store
    empty and a stale ``traffic_laws`` collection behind.

It now:
  1. drops stale collections left over from earlier projects,
  2. builds a richer evidence document per claim variation (with metadata that
     the retriever and the evaluation harness rely on: ``event_id``, category,
     date, language style, source),
  3. chunks long content with overlap,
  4. de-duplicates by content hash, and
  5. writes into the single canonical collection from ``src.config``.

Run from the project root:  ``python -m scripts.update_rag``
"""
import hashlib
import json
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # older langchain layout
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import (
    VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME,
    STALE_COLLECTIONS, CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = "data/dataset.json"


def drop_stale_collections() -> None:
    """Remove collections from prior projects so the store stays clean."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        existing = {c.name for c in client.list_collections()}
        for name in STALE_COLLECTIONS:
            if name in existing:
                client.delete_collection(name)
                logger.info("Dropped stale collection: %s", name)
    except Exception as exc:
        logger.warning("Could not clean stale collections (%s); continuing.", exc)


def build_documents(dataset) -> list:
    """Turn the dataset into retrievable evidence documents with metadata."""
    docs = []
    for entry in dataset:
        event_id = entry.get("claim_id") or entry.get("core_event", "")[:40]
        category = entry.get("category", "general")
        date = entry.get("date", "N/A")
        core_event = entry.get("core_event", "")

        # An event-level summary document.
        if core_event:
            docs.append(Document(
                page_content=f"Event: {core_event}",
                metadata={"source": "local_dataset", "event_id": event_id,
                          "category": category, "date": date, "kind": "event_summary"},
            ))

        # One evidence document per normalized claim variation.
        for var in entry.get("variations", []):
            claim = var.get("normalized_claim", "").strip()
            if not claim:
                continue
            docs.append(Document(
                page_content=claim,
                metadata={"source": "local_dataset", "event_id": event_id,
                          "category": category, "date": date,
                          "language_style": var.get("language_style", "unknown"),
                          "kind": "evidence"},
            ))
    return docs


def dedup_and_chunk(docs) -> list:
    """Split long documents and drop exact-duplicate chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    seen, unique = set(), []
    for c in chunks:
        h = hashlib.md5(c.page_content.strip().lower().encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        unique.append(c)
    return unique


def update_db() -> None:
    drop_stale_collections()

    if not os.path.exists(DATASET_PATH):
        logger.error("%s not found. Nothing to ingest.", DATASET_PATH)
        return
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    docs = build_documents(dataset)
    chunks = dedup_and_chunk(docs)
    logger.info("Prepared %d unique chunks from %d source records.", len(chunks), len(dataset))

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
    )
    if chunks:
        vector_db.add_documents(chunks)
        logger.info("Ingested %d chunks into '%s'.", len(chunks), CHROMA_COLLECTION_NAME)
    else:
        logger.warning("No chunks to ingest.")


if __name__ == "__main__":
    update_db()
