import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

PDF_PATH = "../data/Motor-Traffic-Consolidated-2024.pdf"
DB_DIR = "../vector_db"

def build_database():
    print("1. Loading Sri Lankan Motor Traffic Act PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print("2. Chunking legal text...")
    # Legal texts need careful chunking. 1000 chars with 200 overlap ensures 
    # clauses aren't cut off abruptly.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    print("3. Generating embeddings and building Vector DB...")
    # Using a fast, lightweight open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"✓ Vector database successfully built and saved to {DB_DIR}")

if __name__ == "__main__":
    build_database()