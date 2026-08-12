"""Inspect the current contents of the vector store.

Lists collections, source distribution (Telegram vs RSS feeds),
total document count, and the most recently ingested news articles.

Run:  python -m scripts.inspect_db
"""
import os
import sys
from collections import Counter

# Enforce UTF-8 output on Windows terminals to display Sinhala and Tamil text cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

try:
    import chromadb
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    print("=== ChromaDB Collections ===")
    for c in client.list_collections():
        print(f" - {c.name} ({c.count()} items)")
except Exception as exc:
    print(f"Could not list collections directly: {exc}")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"local_files_only": True}
)
vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION_NAME,
)

print(f"\n=== Inspecting Canonical Collection: '{CHROMA_COLLECTION_NAME}' ===")
data = vector_db.get(include=["documents", "metadatas"])
docs = data.get("documents", [])
metas = data.get("metadatas", [])

print(f"Total Stored Documents: {len(docs)}")

# Calculate source breakdown
sources = Counter(m.get("source", "Unknown") for m in metas if m)
print("\n--- Source Breakdown ---")
for source, count in sources.most_common():
    print(f" • {source:<30}: {count} articles")

if docs:
    print("\n--- 5 Most Recent Ingested Entries ---")
    for i in range(min(5, len(docs))):
        idx = len(docs) - 1 - i
        content_preview = docs[idx].replace('\n', ' ')[:150]
        meta = metas[idx] if idx < len(metas) else {}
        print(f"\n[#{idx + 1}] Source: {meta.get('source', 'Unknown')} | Timestamp: {meta.get('timestamp', 'N/A')}")
        print(f"    Content: {content_preview}...")
