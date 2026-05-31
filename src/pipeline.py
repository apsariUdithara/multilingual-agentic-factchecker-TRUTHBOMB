from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.triage import triage_agent
from src.agents.traffic_expert import traffic_expert_agent
from src.agents.web_researcher import web_researcher_agent
from src.agents.synthesis import synthesis_agent

def router(state: AgentState):
    """
    Determines which expert node to route to based on triage output.
    """
    return state.get("routing_destination", "web_researcher")

# Initialize the Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("triage", triage_agent)
workflow.add_node("traffic_expert", traffic_expert_agent)
workflow.add_node("web_researcher", web_researcher_agent)
workflow.add_node("synthesis", synthesis_agent)

# Set Entry Point
workflow.set_entry_point("triage")

# Add Conditional Edges
workflow.add_conditional_edges(
    "triage",
    router,
    {
        "traffic_expert": "traffic_expert",
        "web_researcher": "web_researcher"
    }
)

# Add Edges to Synthesis
workflow.add_edge("traffic_expert", "synthesis")
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
        "agent_outputs": [],
        "language": "English",
        "extracted_claim": "",
        "routing_destination": "",
        "final_verdict": {}
    }
    
    result = app.invoke(initial_state)
    return result
