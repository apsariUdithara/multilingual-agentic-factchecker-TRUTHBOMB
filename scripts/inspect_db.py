"""Inspect the current contents of the vector store.

Lists every collection (so stray/stale ones are obvious), the document count of
the canonical collection, and a sample entry with its metadata.

Run:  ``python -m scripts.inspect_db``
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

# Show all collections present (helps catch stale ones like traffic_laws).
try:
    import chromadb
    client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
    print("Collections in store:", [c.name for c in client.list_collections()])
except Exception as exc:
    print(f"Could not list collections directly: {exc}")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION_NAME,
)

print(f"\nInspecting collection: {CHROMA_COLLECTION_NAME}")
data = vector_db.get(include=["documents", "metadatas"])
docs = data.get("documents", [])
print(f"Total documents: {len(docs)}")

if docs:
    print("\nSample entry:")
    print(f"Content: {docs[0][:200]}...")
    print(f"Metadata: {data['metadatas'][0]}")
