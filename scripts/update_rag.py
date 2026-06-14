import json
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

def update_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )

    # Example: Loading from a new dataset
    try:
        with open("data/dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        docs = []
        for entry in data:
            doc = Document(
                page_content=entry.get("core_event", ""),
                metadata={"source": "local_json", "type": entry.get("category", "general")}
            )
            docs.append(doc)
        
        if docs:
            vector_db.add_documents(docs)
            print(f"Added {len(docs)} documents to {CHROMA_COLLECTION_NAME}")
    except FileNotFoundError:
        print("data/dataset.json not found. Skipping initial ingest.")

if __name__ == "__main__":
    update_db()
