from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.claim_extractor import claim_extractor_agent
from src.agents.local_researcher import local_researcher_agent
from src.agents.web_researcher import web_researcher_agent
from src.agents.synthesis import synthesis_agent


def router(state: AgentState) -> str:
    """
    Interim deterministic routing (local-first).

    The claim extractor no longer chooses a route (Phase 2). Until the dedicated
    retrieve node (Phase 3) and the score gate (Phase 4) are in place, we default
    to the local researcher and treat any unknown value as local, so the graph
    always has a valid edge to follow.
    """
    dest = state.get("routing_destination") or "local_researcher"
    return dest if dest in ("local_researcher", "web_researcher") else "local_researcher"


# Initialize the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("claim_extractor", claim_extractor_agent)
workflow.add_node("local_researcher", local_researcher_agent)
workflow.add_node("web_researcher", web_researcher_agent)
workflow.add_node("synthesis", synthesis_agent)

# Set Entry Point
workflow.set_entry_point("claim_extractor")

# Add Conditional Edges
workflow.add_conditional_edges(
    "claim_extractor",
    router,
    {
        "local_researcher": "local_researcher",
        "web_researcher": "web_researcher",
    },
)

# Add Edges to Synthesis
workflow.add_edge("local_researcher", "synthesis")
workflow.add_edge("web_researcher", "synthesis")

# End at Synthesis
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
        "routing_destination": "local_researcher",  # interim local-first default (Phase 4 -> score-based)
        "agent_outputs": [],
        "retrieved_docs": [],   # populated by the retrieve node (Phase 3)
        "retrieval_score": 0.0,  # populated by the retrieve node; drives routing (Phase 4)
        "audit_trail": [],
        "final_verdict": {},
    }

    result = app.invoke(initial_state)
    return result