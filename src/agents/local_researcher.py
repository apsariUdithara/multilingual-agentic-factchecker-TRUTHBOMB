from src.state import AgentState
from src.tools.local_db_tool import verify_claim_against_local_db, get_retriever

def local_researcher_agent(state: AgentState) -> dict:
    """
    Local verify node.
    Extracts document proofs, verifies with GGUF model, and falls back to live web search
    if local OSINT DB lacks evidence.
    """
    claim = state.get("extracted_claim", "")

    try:
        # Extract raw document proofs with exact metadata (Source name, Timestamp)
        retrieved_items = get_retriever().retrieve(claim)
        doc_proofs = []
        for item in retrieved_items:
            doc = item.get("doc")
            if doc:
                doc_proofs.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "Local DB News Feed") if doc.metadata else "Local DB News Feed",
                    "timestamp": doc.metadata.get("timestamp", doc.metadata.get("ingested_at", "Just now")) if doc.metadata else "Just now",
                    "distance_score": float(item.get("score", 0.0))
                })

        # Invoke local verification model
        verification_summary = verify_claim_against_local_db.invoke({"claim": claim})
        
        current_outputs = state.get("agent_outputs", [])
        new_output = {
            "agent": "local_researcher",
            "findings": verification_summary,
            "source_type": "local_osint_db",
            "retrieved_docs": doc_proofs
        }
        current_outputs.append(new_output)

        # Check if local DB retrieved matching evidence; if not, fall back automatically to live web search
        has_matching_proof = False
        claim_keywords = [w.lower() for w in claim.split() if len(w) > 3]
        for p in doc_proofs:
            content_lower = p["content"].lower()
            if sum(1 for kw in claim_keywords if kw in content_lower) >= max(1, len(claim_keywords) // 2):
                has_matching_proof = True
                break

        if not has_matching_proof:
            try:
                from src.agents.web_researcher import web_researcher_agent
                web_result = web_researcher_agent(state)
                web_outputs = web_result.get("agent_outputs", [])
                for wo in web_outputs:
                    if wo not in current_outputs:
                        current_outputs.append(wo)
            except Exception as web_err:
                print(f"Fallback web search error: {web_err}")

        return {
            "agent_outputs": current_outputs,
            "retrieved_docs": doc_proofs,
            "audit_trail": [
                {"node": "local_researcher", "status": "ok", "detail": verification_summary[:160]}
            ]
        }
    except Exception as e:
        print(f"Error in local_researcher_agent: {e}")
        return {
            "agent_outputs": state.get("agent_outputs", []) + [{
                "agent": "local_researcher",
                "findings": f"Failed to perform local database research: {e}",
                "source_type": "error"
            }],
            "audit_trail": [
                {"node": "local_researcher", "status": "error", "detail": str(e)[:160]}
            ]
        }