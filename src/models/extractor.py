"""
Local claim extractor backed by the fine-tuned Phi-3.5-mini GGUF model.

Runs via llama.cpp at temperature 0 with a GBNF grammar that constrains the
output to a strict JSON object: {"language": <enum>, "extracted_claim": <string>}.
Grammar-constraining is what makes this node deterministic and guaranteed
parseable -- the model cannot emit prose, markdown, or malformed JSON.

The model is exposed through a lazy singleton (`get_extractor`) so the (heavy)
GGUF file is loaded once on first use rather than at import time. This keeps
`src.pipeline` importable and testable even when the model file is absent.
"""

import os
import json

from llama_cpp import Llama, LlamaGrammar

from src.config import GGUF_MODEL_PATH

# GBNF grammar: forces exactly {"language": <one enum>, "extracted_claim": <json string>}.
# The `string`/`char` rules are the canonical llama.cpp JSON-string definition.
_EXTRACTION_GRAMMAR = r'''
root   ::= "{" ws "\"language\"" ws ":" ws lang ws "," ws "\"extracted_claim\"" ws ":" ws string ws "}"
lang   ::= "\"English\"" | "\"Sinhala\"" | "\"Tamil\"" | "\"Singlish\""
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
ws     ::= [ \t\n]*
'''

_INSTRUCTION = (
    "You are an expert linguistic AI for a Sri Lankan fact-checking system. "
    "Identify the primary language of the post (English, Sinhala, Tamil, or Singlish) "
    "and extract the single core verifiable claim, normalized into professional English "
    "with all emotion and slang removed."
)


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

    def extract(self, raw_post: str) -> dict:
        """
        Extract the language + core claim as a validated dict:
            {"language": str, "extracted_claim": str}
        """
        prompt = (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n" + _INSTRUCTION + "\n\n"
            f"### Input:\n{raw_post}\n\n"
            "### Response:\n"
        )

        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.0,   # deterministic extraction
            top_p=1.0,
            grammar=self.grammar,
            echo=False,
            stop=["<|end|>", "<|endoftext|>", "### Instruction:"],
        )

        raw_text = response["choices"][0]["text"].strip()
        # The grammar guarantees valid JSON of the expected shape.
        data = json.loads(raw_text)
        return {
            "language": data.get("language", "English"),
            "extracted_claim": (data.get("extracted_claim") or "").strip(),
        }

    def extract_claim(self, raw_post: str) -> str:
        """Backward-compatible helper returning only the claim string."""
        return self.extract(raw_post)["extracted_claim"]


# --- Lazy singleton --------------------------------------------------------
_extractor_instance = None


def get_extractor() -> LinguisticExtractor:
    """Return the shared extractor, loading the GGUF model on first call."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LinguisticExtractor()
    return _extractor_instance