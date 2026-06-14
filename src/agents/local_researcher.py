from src.state import AgentState
from src.tools.local_db_tool import verify_claim_against_local_db

def local_researcher_agent(state: AgentState) -> dict:
    """
    Uses the local vector DB (OSINT context) to verify the extracted claim.
    """
    claim = state.get("extracted_claim", "")
    
    try:
        # Invoke the local tool
        verification_summary = verify_claim_against_local_db.invoke({"claim": claim})
        
        # Update agent_outputs
        current_outputs = state.get("agent_outputs", [])
        new_output = {
            "agent": "local_researcher",
            "findings": verification_summary,
            "source_type": "local_osint_db"
        }
        
        return {
            "agent_outputs": current_outputs + [new_output]
        }
    except Exception as e:
        print(f"Error in local_researcher_agent: {e}")
        return {
            "agent_outputs": state.get("agent_outputs", []) + [{
                "agent": "local_researcher",
                "findings": f"Failed to perform local database research: {e}",
                "source_type": "error"
            }]
        }
