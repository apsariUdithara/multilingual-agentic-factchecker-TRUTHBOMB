from src.state import AgentState
from src.models.extractor import get_extractor


def claim_extractor_agent(state: AgentState) -> dict:
    """
    Deterministically extract the core verifiable claim and its language from the
    raw input using the local Phi-3 GGUF model with GBNF-constrained JSON output.

    Routing is intentionally NOT decided here. Per the target architecture, the
    local-vs-web decision is a separate, score-based step taken after evidence
    retrieval (Phases 3-4), not an LLM guess made during extraction.
    """
    raw_input = state.get("raw_input", "")

    try:
        result = get_extractor().extract(raw_input)
        language = result.get("language", "English")
        extracted_claim = (result.get("extracted_claim") or "").strip() or raw_input
        status = "ok"
        detail = f"Claim extracted via Phi-3 GGUF + GBNF (language: {language})."
    except Exception as e:
        print(f"Error in claim_extractor_agent: {e}")
        language = "English"
        extracted_claim = raw_input
        status = "fallback"
        detail = f"Extraction failed ({e}); using raw input as claim."

    return {
        "language": language,
        "extracted_claim": extracted_claim,
        "audit_trail": [
            {"node": "claim_extractor", "status": status, "detail": detail}
        ],
    }