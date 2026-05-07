from unsloth import FastLanguageModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

DB_DIR = "../vector_db"
MODEL_DIR = "../models/phi3-traffic-lora"

print("1. Initializing Vector Database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

print("2. Loading Fine-Tuned Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Phi-3.5-mini-instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
# Apply your trained LoRA weights
model.load_adapter(MODEL_DIR)
FastLanguageModel.for_inference(model)

# The updated prompt template that includes LEGAL CONTEXT
rag_prompt_template = """<|system|>
You are an expert multilingual fact-checker for Sri Lankan traffic law. Analyze the claim using ONLY the provided legal context. Think step-by-step, and output a JSON-like format with Verdict, Uncertainty, and Rationale.<|end|>
<|user|>
Legal Context:
{context}

Claim: {input}<|end|>
<|assistant|>
"""

def check_traffic_claim(claim):
    print(f"\nAnalyzing Claim: '{claim}'")
    
    # Step A: Retrieve relevant laws from the PDF
    docs = vector_db.similarity_search(claim, k=3)
    retrieved_context = "\n---\n".join([doc.page_content for doc in docs])
    
    # Step B: Format the prompt
    formatted_prompt = rag_prompt_template.format(
        context=retrieved_context,
        input=claim
    )
    
    # Step C: Generate Inference
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Clean output
    assistant_start = response.rfind("<|assistant|>") + len("<|assistant|>")
    final_output = response[assistant_start:].strip()
    
    print("\n--- SYSTEM RESPONSE ---")
    print(final_output)
    print("-----------------------\n")

# --- Testing the System ---
if __name__ == "__main__":
    test_claims = [
        "A 16-year-old can be registered as the owner of a motor vehicle.",
        "Emergency vehicles like fire engines and police cars on pursuit must obey standard speed limits.",
        "A driver is allowed to use a mobile phone while driving if they are careful."
    ]
    
    for claim in test_claims:
        check_traffic_claim(claim)