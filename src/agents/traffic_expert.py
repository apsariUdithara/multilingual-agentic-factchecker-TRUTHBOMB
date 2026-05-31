from src.state import AgentState
from src.tools.traffic_db_tool import verify_traffic_law_claim

def traffic_expert_agent(state: AgentState) -> dict:
    """
    Uses the Phi-3.5-mini LoRA model and local vector DB to verify traffic claims.
    """
    claim = state.get("extracted_claim", "")
    
    try:
        # Invoke the local tool
        verification_summary = verify_traffic_law_claim.invoke({"claim": claim})
        
        # Update agent_outputs
        current_outputs = state.get("agent_outputs", [])
        new_output = {
            "agent": "traffic_expert",
            "findings": verification_summary,
            "source_type": "local_traffic_db"
        }
        
        return {
            "agent_outputs": current_outputs + [new_output]
        }
    except Exception as e:
        print(f"Error in traffic_expert_agent: {e}")
        return {
            "agent_outputs": state.get("agent_outputs", []) + [{
                "agent": "traffic_expert",
                "findings": f"Failed to perform traffic law research locally: {e}",
                "source_type": "error"
            }]
        }
