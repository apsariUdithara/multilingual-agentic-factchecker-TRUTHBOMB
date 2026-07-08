"""
Local claim extractor backed by the fine-tuned Phi-3.5-mini GGUF model.

Runs via llama.cpp at temperature 0 with a GBNF grammar that constrains the
output to a strict JSON object: {"language": <enum>, "extracted_claim": <string>}.
Grammar-constraining is what makes this node deterministic and guaranteed
parseable -- the model cannot emit prose, markdown, or malformed JSON.

The model is exposed through a lazy singleton (`get_extractor`) so the (heavy)
GGUF file is loaded once on first use rather than at import time. `get_llm`
exposes that same loaded model so the local verifier can reuse it (one GGUF in
VRAM, shared between extraction and local verification).
"""

import os
import re
from llama_cpp import Llama
from src.config import GGUF_MODEL_PATH

class LinguisticExtractor:
    """Wraps the Phi-3 GGUF model and returns structured, grammar-constrained claims."""

    def __init__(self, model_path: str = GGUF_MODEL_PATH):
        print("Initializing LinguisticExtractor via llama.cpp (GPU accelerated)...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model file not found at '{model_path}'.")

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            verbose=False,
        )
        self.grammar = LlamaGrammar.from_string(_EXTRACTION_GRAMMAR)

    def extract_claim(self, raw_post: str) -> str:
        # Few-Shot In-Context Prompting to anchor the model's extraction capabilities
        # and prevent it from hallucinating memorized training samples on noisy/garbled inputs.
        prompt = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. "
            "Translate it to professional English, remove all emotion and slang, and output only the declarative claim.\n\n"
            "### Example 1 Input:\n"
            "Highway eka ada wahalada? Kadawatha gawa loku accident ekak wela kiyala aranchiyak awa. Kauda danne?\n\n"
            "### Example 1 Response:\n"
            "The highway is closed today due to a major accident near Kadawatha.\n\n"
            "### Example 2 Input:\n"
            "Breaking: Central bank eken interest rates drop karanna decision ekak aran thiyenne next month indan.\n\n"
            "### Example 2 Response:\n"
            "The Central Bank has decided to lower interest rates starting next month.\n\n"
            "### Example 3 Input:\n"
            "6 මැණික්හින්ඉන් 'මුතූ' සිව්වැනි බිල්ලත් ගනී කද ள் SN වට්ටම් டட තැකී කෑම දන්න ගිහින් ජීවිතයක් අහිමි වයි\n\n"
            "### Example 3 Response:\n"
            "'Muthu' of Menikhinna claims its fourth victim; a life is lost while trying to offer food.\n\n"
            f"### Input:\n{raw_post}\n\n"
            "### Response:\n"
        )

        # Decoder Parameter Tuning:
        # Introducing a low temperature and repeat penalty prevents the model from getting stuck
        # on deterministic training sample sequence loops when faced with noisy text.
        response = self.llm(
            prompt,
            max_tokens=128,
            temperature=0.2, 
            top_p=0.9,
            repeat_penalty=1.15,
            echo=False,
            stop=["<|end|>", "<|endoftext|>", "### Instruction:", "### Input:"]
        )
        
        extracted_text = response["choices"][0]["text"].strip()

        # Step 3: Check for suspected hallucinations and route to Gemini API fallback if needed
        if is_suspected_hallucination(raw_post, extracted_text):
            print("WARNING: Suspected hallucination detected in local extraction. Falling back to Gemini API...")
            try:
                import google.generativeai as genai
                from src.config import GEMINI_API_KEY
                
                if GEMINI_API_KEY:
                    genai.configure(api_key=GEMINI_API_KEY)
                    gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
                    
                    gemini_prompt = (
                        "You are an expert linguistic AI. Extract the core verifiable claim from the following Sri Lankan social media post. "
                        "Translate it to professional English, remove all emotion and slang, and output only the declarative claim.\n\n"
                        f"Input: {raw_post}"
                    )
                    gemini_response = gemini_model.generate_content(gemini_prompt)
                    fallback_claim = gemini_response.text.strip()
                    if fallback_claim:
                        print(f"Fallback successful. Gemini extracted claim: {fallback_claim}")
                        return fallback_claim
            except Exception as e:
                print(f"Gemini fallback failed: {e}")
                
        return extracted_text