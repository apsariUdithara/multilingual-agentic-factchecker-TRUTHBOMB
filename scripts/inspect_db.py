from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

# Use configuration from src.config
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION_NAME
)

print(f"Inspecting collection: {CHROMA_COLLECTION_NAME}")
data = vector_db.get(include=["documents", "metadatas"])
print(f"Total documents: {len(data['documents'])}")

if len(data['documents']) > 0:
    print("\nSample entry:")
    print(f"Content: {data['documents'][0][:200]}...")
    print(f"Metadata: {data['metadatas'][0]}")
