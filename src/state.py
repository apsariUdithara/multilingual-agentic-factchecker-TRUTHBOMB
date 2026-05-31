from typing import Annotated, List, TypedDict, Union, Dict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    raw_input: str  # text or image path
    language: str   # English, Sinhala, Tamil, or Singlish
    extracted_claim: str
    routing_destination: str  # 'traffic_expert' or 'web_researcher'
    agent_outputs: List[Dict] # Evidence gathered
    final_verdict: Dict       # { "verdict": str, "explanation": str, "language": str }
