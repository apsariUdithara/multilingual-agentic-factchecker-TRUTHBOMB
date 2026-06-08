from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_DIR = "D:/7th/AI/Project/chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_COLLECTION_NAME = "traffic_telemetry"

print("Loading vector database to inspect ingested content...")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION_NAME
)

# Fetch data from the database
db_content = vector_db.get(include=["documents", "metadatas"])
docs = db_content.get("documents", [])
metadatas = db_content.get("metadatas", [])

print(f"\nTotal items found in collection '{CHROMA_COLLECTION_NAME}': {len(docs)}\n")
print("=" * 70)
print("SHOWING INGESTED TELEMETRY CHUNKS:")
print("=" * 70)

for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
    print(f"\n[Record #{i}]")
    print(f"Source ID: {meta.get('message_id')}")
    print(f"Timestamp: {meta.get('timestamp')}")
    print(f"Content:\n{doc}")
    print("-" * 50)