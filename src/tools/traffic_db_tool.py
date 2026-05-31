import torch
from langchain.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.config import TRAFFIC_MODEL_PATH, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

# Global variables for lazy loading
_embeddings = None
_vector_db = None
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
        
        # Load LoRA adapters
        _model = PeftModel.from_pretrained(base_model, TRAFFIC_MODEL_PATH)
        _model.eval()
    return _model, _tokenizer

@tool
def verify_traffic_law_claim(claim: str) -> str:
    """
    Verifies a traffic law claim using the local Sri Lankan Motor Traffic Act database 
    and a fine-tuned Phi-3.5 model.
    """
    db = get_vector_db()
    docs = db.similarity_search(claim, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    model, tokenizer = get_model_and_tokenizer()
    
    prompt = f"""<|system|>
You are a Sri Lankan Traffic Law Expert. Use the following context to verify the claim.
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
    # Extract only the assistant's part
    if "<|assistant|>" in response:
        response = response.split("<|assistant|>")[-1].strip()
        
    return response
