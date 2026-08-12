import operator
from typing import Annotated, List, TypedDict, Dict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    raw_input: str            # text or image path
    language: str             # English, Sinhala, Tamil, or Singlish
    extracted_claim: str
    extracted_claim_a: str  # Debater A: Fine-Tuned Local GGUF Model
    extracted_claim_b: str  # Debater B: Cloud Gemini LLM
    consensus_status: str   # "AGREEMENT" or "DISAGREEMENT"
    routing_destination: str  # 'local_researcher' or 'web_researcher'
    agent_outputs: List[Dict]                          # evidence gathered (manual concat in agents)
    retrieved_docs: List[Dict]                         # (Phase 3) structured evidence from retrieve node
    retrieval_score: float                             # (Phase 4) confidence used for score-based routing
    audit_trail: Annotated[List[Dict], operator.add]   # per-node trace; auto-merged across nodes
    final_verdict: Dict       # { "verdict": str, "confidence_score": str, "rationale": str, "summaries": {...} }
