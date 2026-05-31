from typing import Annotated, List, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[dict], add_messages]
    claim: str
    category: str  # 'traffic' or 'general'
    verification_results: List[dict]
    final_verdict: str
