from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.config import ROUTING_SCORE_THRESHOLD
from src.agents.claim_extractor import claim_extractor_agent
from src.agents.retriever import retrieve_node
from src.agents.local_researcher import local_researcher_agent
from src.agents.web_researcher import web_researcher_agent
from src.agents.synthesis import synthesis_agent


def router(state: AgentState) -> str:
    """
    Score-based routing (deterministic, auditable).

    Local-first: if the best local evidence meets ROUTING_SCORE_THRESHOLD, verify
    against the local OSINT knowledge base; otherwise escalate to trusted web search.
    The score is `retrieval_score` (best dense similarity in the KB) set by the
    retrieve node.
    """
    score = state.get("retrieval_score", 0.0) or 0.0
    return "local_researcher" if score >= ROUTING_SCORE_THRESHOLD else "web_researcher"


# Initialize the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("claim_extractor", claim_extractor_agent)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("local_researcher", local_researcher_agent)
workflow.add_node("web_researcher", web_researcher_agent)
workflow.add_node("synthesis", synthesis_agent)

# Set Entry Point
workflow.set_entry_point("claim_extractor")

# Extract -> Retrieve -> (score-based route)
workflow.add_edge("claim_extractor", "retrieve")
workflow.add_conditional_edges(
    "retrieve",
    router,
    {
        "local_researcher": "local_researcher",
        "web_researcher": "web_researcher",
    },
)

# Researchers -> Synthesis -> END
workflow.add_edge("local_researcher", "synthesis")
workflow.add_edge("web_researcher", "synthesis")
workflow.add_edge("synthesis", END)

# Compile
app = workflow.compile()


def run_fact_checker(input_text: str):
    """
    Executes the fact-checking pipeline for a given input.
    """
    initial_state = {
        "raw_input": input_text,
        "messages": [],
        "language": "English",
        "extracted_claim": "",
        "routing_destination": "local_researcher",  # seeded; router now decides by score
        "agent_outputs": [],
        "retrieved_docs": [],
        "retrieval_score": 0.0,
        "audit_trail": [],
        "final_verdict": {},
    }

    result = app.invoke(initial_state)
    return result