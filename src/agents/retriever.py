from src.state import AgentState
from src.tools.hybrid_retriever import get_hybrid_retriever


def retrieve_node(state: AgentState) -> dict:
    """
    Hybrid retrieve node (Phase 3).

    Runs dense + lexical retrieval, fuses them with RRF, and re-ranks by source
    tier and recency. Populates `retrieved_docs` (structured, with per-document
    component scores) and `retrieval_score` (best semantic match in the KB), which
    the router will threshold for local-vs-web escalation in Phase 4.
    """
    claim = state.get("extracted_claim", "")

    try:
        docs, score = get_hybrid_retriever().retrieve(claim)
        status = "ok"
        detail = f"Retrieved {len(docs)} docs after re-ranking; best match {score:.2f}."
    except Exception as e:
        print(f"Error in retrieve_node: {e}")
        docs, score = [], 0.0
        status = "error"
        detail = f"Retrieval failed ({e})."

    return {
        "retrieved_docs": docs,
        "retrieval_score": score,
        "audit_trail": [{"node": "retrieve", "status": status, "detail": detail}],
    }