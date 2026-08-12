from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from llama_cpp import Llama
from src.config import (
    GGUF_MODEL_PATH, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME,
)
from src.rag.retriever import HybridRetriever

# Global variables for lazy loading
_embeddings = None
_vector_db = None
_retriever = None
_llm = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"local_files_only": True}
            )
        except Exception:
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
    try:
        import src.main as main_module
        if main_module.retriever is not None:
            return main_module.retriever
    except Exception:
        pass

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

def get_llama_model():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=GGUF_MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False
        )
    return _llm

@tool
def verify_claim_against_local_db(claim: str) -> str:
    """
    Verifies a social media claim using the local OSINT database
    and a fine-tuned reasoning model.
    """
    # Same hybrid retrieval the API uses: dense + sparse + RRF + rerank.
    retrieved = get_retriever().retrieve(claim)
    
    if not retrieved:
        return "No relevant context found in local database."

    context_docs = []
    for idx, item in enumerate(retrieved[:3]):
        source = item['doc'].metadata.get('source', 'Unknown')
        content = item['doc'].page_content.strip()
        context_docs.append(f"[Document {idx+1} | Source: {source}]\n{content}")
        
    context_blob = "\n\n".join(context_docs)

    # Reuse the existing GGUF model instance already loaded in memory to prevent VRAM OOM / file locking
    llm = None
    try:
        import src.main as main_module
        if main_module.extractor_model is not None and hasattr(main_module.extractor_model, "llm"):
            llm = main_module.extractor_model.llm
    except Exception:
        pass

    if llm is None:
        # Fallback if running outside the main API process
        llm = get_llama_model()
    
    prompt = f"""<|system|>
You are a factual Sri Lankan news analyst. 
Analyze whether the retrieved context contains evidence directly supporting or refuting the claim.
Rules:
- If the context contains news about the exact incident, summarize the facts accurately.
- Do NOT confuse different incidents in different locations. If context describes a completely different location or event, state that context does not match.
Context:
{context_blob}
<|user|>
Claim: {claim}
<|assistant|>
"""
    
    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.0,
        top_p=0.9,
        echo=False
    )
    
    llm_analysis = response["choices"][0]["text"].strip()
    return f"--- Retrieved Ground-Truth Documents ---\n{context_blob}\n\n--- Local Model Analysis ---\n{llm_analysis}"
