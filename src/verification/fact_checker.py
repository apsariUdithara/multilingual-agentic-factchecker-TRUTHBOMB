import json
import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from llama_cpp import Llama
from src.config import GGUF_MODEL_PATH, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, CHROMA_COLLECTION_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactChecker:
    def __init__(
        self,
        model_path: str = GGUF_MODEL_PATH,
        vector_db_dir: str = VECTOR_DB_DIR,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        collection_name: str = CHROMA_COLLECTION_NAME
    ):
        """
        Initializes the FactChecker with a local LLM and a vector store.
        """
        logger.info("Initializing FactChecker...")
        
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Initialize Vector Store
        self.vector_db = Chroma(
            persist_directory=vector_db_dir,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        
        # Initialize Local LLM via llama.cpp
        # Using n_gpu_layers=-1 for full GPU offloading
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False
        )
        logger.info("FactChecker initialized successfully.")

    def verify_claim(self, claim: str) -> Dict[str, Any]:
        """
        Verifies a declarative claim using RAG and returns a structured JSON response.
        """
        try:
            logger.info(f"Verifying claim: {claim}")
            
            # 1. Retrieval: Query the vector store for relevant context
            docs = self.vector_db.similarity_search(claim, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            sources = [doc.metadata.get("source", "Unknown Source") for doc in docs]
            
            # 2. Augmentation & Generation: Prompt the LLM
            prompt = self._build_prompt(claim, context)
            
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.0, # Factual consistency
                stop=["<|end|>", "### Instruction:"]
            )
            
            raw_output = response["choices"][0]["text"].strip()
            
            # 3. Parsing the structured output
            return self._parse_response(raw_output, sources)

        except Exception as e:
            logger.error(f"Error during claim verification: {e}")
            return {
                "is_verified": False,
                "reasoning": f"An error occurred during verification: {str(e)}",
                "sources": []
            }

    def _build_prompt(self, claim: str, context: str) -> str:
        """
        Constructs the prompt for the LLM to perform verification.
        """
        return (
            "Below is a claim and some context retrieved from a Sri Lankan OSINT news/social-media archive. "
            "Your task is to verify if the claim is supported or refuted by the context. "
            "Output your response strictly in JSON format with the following keys: "
            "'is_verified' (boolean), 'reasoning' (string), and 'sources' (list of strings).\n\n"
            "### Context:\n"
            f"{context}\n\n"
            "### Claim:\n"
            f"{claim}\n\n"
            "### Response (JSON):\n"
        )

    def _parse_response(self, raw_output: str, default_sources: List[str]) -> Dict[str, Any]:
        """
        Attempts to parse the LLM's raw output as JSON.
        """
        try:
            # Look for JSON structure in case the model added extra text
            start_idx = raw_output.find('{')
            end_idx = raw_output.rfind('}') + 1
            if start_idx != -1 and end_idx != -1:
                json_str = raw_output[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                data = json.loads(raw_output)
            
            # Ensure sources are populated if the LLM didn't provide them
            if not data.get("sources"):
                data["sources"] = list(set(default_sources))
                
            return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse LLM output as JSON: {e}. Raw output: {raw_output}")
            return {
                "is_verified": False,
                "reasoning": f"Failed to parse verification response. Raw output: {raw_output}",
                "sources": list(set(default_sources))
            }

def verify_claim(claim: str) -> Dict[str, Any]:
    """
    Convenience function to instantiate and run the fact checker.
    Note: In a production environment, you'd want to reuse the FactChecker instance.
    """
    checker = FactChecker()
    return checker.verify_claim(claim)
