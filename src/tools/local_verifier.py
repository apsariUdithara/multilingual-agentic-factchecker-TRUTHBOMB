"""
Local verifier (Phase 4).

Judges a claim against the evidence already gathered by the retrieve node
(`retrieved_docs`) using the shared Phi-3 GGUF model at temperature 0, under a
GBNF grammar that constrains output to {stance, confidence, reasoning}.

The verifier does NOT retrieve -- it consumes the re-ranked documents produced
upstream, so retrieval happens exactly once per request.

stance is one of SUPPORTED / REFUTED / INSUFFICIENT. This is an *evidence*
stance, not the final verdict; the adjudicator (Phase 5) maps stances from the
local and web verifiers onto the final verdict vocabulary.
"""

import json
from typing import List, Dict

from llama_cpp import LlamaGrammar

from src.models.extractor import get_llm

_VERDICT_GRAMMAR = r'''
root   ::= "{" ws "\"stance\"" ws ":" ws stance ws "," ws "\"confidence\"" ws ":" ws number ws "," ws "\"reasoning\"" ws ":" ws string ws "}"
stance ::= "\"SUPPORTED\"" | "\"REFUTED\"" | "\"INSUFFICIENT\""
number ::= digit ( digit ( digit )? )?
digit  ::= [0-9]
string ::= "\"" char* "\""
char   ::= [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
ws     ::= [ \t\n]*
'''

_grammar = None


def _get_grammar() -> LlamaGrammar:
    global _grammar
    if _grammar is None:
        _grammar = LlamaGrammar.from_string(_VERDICT_GRAMMAR)
    return _grammar


def _build_context(docs: List[Dict]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        source = d.get("source", "unknown")
        tier = d.get("tier", "")
        timestamp = d.get("timestamp", "")
        content = d.get("content", "")
        lines.append(f"[{i}] ({source}, {tier}, {timestamp})\n{content}")
    return "\n\n".join(lines)


def verify_locally(claim: str, retrieved_docs: List[Dict]) -> Dict:
    """
    Return {"stance": str, "confidence": int (0-100), "reasoning": str}.

    Short-circuits to INSUFFICIENT (no model call) when there is no local evidence.
    """
    if not retrieved_docs:
        return {
            "stance": "INSUFFICIENT",
            "confidence": 0,
            "reasoning": "No local evidence was retrieved for this claim.",
        }

    context = _build_context(retrieved_docs)
    prompt = (
        "<|system|>\n"
        "You are a meticulous fact-checking analyst for Sri Lankan news. Judge the claim "
        "ONLY against the provided evidence. If the evidence does not clearly support or "
        "refute the claim, answer INSUFFICIENT.\n"
        "<|user|>\n"
        f"Claim: {claim}\n\n"
        f"Evidence:\n{context}\n\n"
        "Return a stance (SUPPORTED, REFUTED, or INSUFFICIENT), an integer confidence from "
        "0 to 100, and a one-sentence reasoning.\n"
        "<|assistant|>\n"
    )

    llm = get_llm()
    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.0,
        top_p=1.0,
        grammar=_get_grammar(),
        echo=False,
        stop=["<|end|>", "<|endoftext|>", "<|user|>"],
    )

    data = json.loads(response["choices"][0]["text"].strip())
    confidence = max(0, min(100, int(data.get("confidence", 0))))
    return {
        "stance": data.get("stance", "INSUFFICIENT"),
        "confidence": confidence,
        "reasoning": (data.get("reasoning") or "").strip(),
    }