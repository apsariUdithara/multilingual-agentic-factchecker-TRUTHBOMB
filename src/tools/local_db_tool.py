import torch
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import (
    LORA_ADAPTER_PATH, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME,
)
from src.rag.retriever import HybridRetriever

# Global variables for lazy loading
_embeddings = None
_vector_db = None
_retriever = None
_model = None
_tokenizer = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings

def get_vector_db():
    global _vector_db
    if _vector_db is None:
        _vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=get_embeddings(),
            collection_name=CHROMA_COLLECTION_NAME
        )
    return _vector_db

def get_retriever():
    """Build (once) the same hybrid retriever the API uses, over the local DB."""
    global _retriever
    if _retriever is None:
        db = get_vector_db()
        bm25 = None
        try:
            data = db.get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=t, metadata=m or {})
                for t, m in zip(data.get("documents", []), data.get("metadatas", []))
            ]
            if docs:
                bm25 = BM25Retriever.from_documents(docs)
        except Exception:
            bm25 = None
        _retriever = HybridRetriever(db, bm25_retriever=bm25)
    return _retriever

def get_model_and_tokenizer():
    global _model, _tokenizer
    if _model is None:
        base_model_name = "unsloth/Phi-3.5-mini-instruct"
        _tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        # Load base model in 4-bit for optimization
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Load adapters (PEFT)
        _model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
        _model.eval()
    return _model, _tokenizer

@tool
def verify_claim_against_local_db(claim: str) -> str:
    """
    Verifies a social media claim using the local OSINT database
    and a fine-tuned reasoning model.
    """
    # Same hybrid retrieval the API uses: dense + sparse + RRF + rerank.
    retrieved = get_retriever().retrieve(claim)
    context = "\n".join(item["doc"].page_content for item in retrieved)

    model, tokenizer = get_model_and_tokenizer()
    
    prompt = f"""<|system|>
You are an expert fact-checker specialized in Sri Lankan social media claims. 
Use the following context from official news reports and social media archives to verify the claim.
Context:
{context}
<|user|>
Claim: {claim}
<|assistant|>
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
        
    return response
