import asyncio
from telethon import TelegramClient
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from datetime import datetime
from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

# Credentials
API_ID = 38020229
API_HASH = 'bad58fa93d224c8320db9639efa7ed5e'
SESSION_NAME = 'rag_updater_session'

async def main():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_db = Chroma(
        persist_directory=VECTOR_DB_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )

    client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
    await client.start()

    channels = ['AdaDeranaNews']
    
    for channel in channels:
        async for message in client.iter_messages(channel, limit=10):
            if message.text:
                doc = Document(
                    page_content=message.text,
                    metadata={"source": channel, "timestamp": str(message.date)}
                )
                vector_db.add_documents([doc])
                print(f"Added message from {channel}")

if __name__ == "__main__":
    asyncio.run(main())
