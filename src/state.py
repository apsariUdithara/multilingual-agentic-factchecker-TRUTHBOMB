from typing import Annotated, List, TypedDict, Union, Dict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    raw_input: str  # text or image path
    language: str   # English, Sinhala, Tamil, or Singlish
    extracted_claim: str
    extracted_claim_a: str  # Debater A: Fine-Tuned Local GGUF Model
    extracted_claim_b: str  # Debater B: Cloud Gemini LLM
    consensus_status: str   # "AGREEMENT" or "DISAGREEMENT"
    routing_destination: str  # 'local_researcher' or 'web_researcher'
    agent_outputs: List[Dict] # Evidence gathered
    final_verdict: Dict       # Judge verdict synthesis output
