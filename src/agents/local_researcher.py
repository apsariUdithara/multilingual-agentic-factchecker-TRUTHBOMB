from src.state import AgentState
from src.tools.local_verifier import verify_locally


def local_researcher_agent(state: AgentState) -> dict:
    """
    Local verify node (Phase 4).

    Consumes the re-ranked evidence from the retrieve node (`retrieved_docs`) and
    judges the claim with the local Phi-3 GGUF model under a GBNF grammar. Emits a
    readable `findings` string (for synthesis) plus structured `stance`/`confidence`
    fields (for the Phase 5 adjudicator).
    """
    claim = state.get("extracted_claim", "")
    docs = state.get("retrieved_docs", [])

    try:
        result = verify_locally(claim, docs)
        stance = result["stance"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]
        findings = f"Local stance: {stance} (confidence {confidence}%). {reasoning}"
        source_type = "local_osint_db"
        status = "ok"
    except Exception as e:
        print(f"Error in local_researcher_agent: {e}")
        stance, confidence = "INSUFFICIENT", 0
        reasoning = f"Local verification error: {e}"
        findings = reasoning
        source_type = "error"
        status = "error"

    new_output = {
        "agent": "local_researcher",
        "findings": findings,
        "source_type": source_type,
        "stance": stance,
        "confidence": confidence,
    }

    return {
        "agent_outputs": state.get("agent_outputs", []) + [new_output],
        "audit_trail": [
            {"node": "local_researcher", "status": status, "detail": findings[:160]}
        ],
    }