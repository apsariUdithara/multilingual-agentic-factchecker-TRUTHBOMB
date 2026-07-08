import json
from typing import List

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.config import GEMINI_API_KEY
from src.state import AgentState
from src.tools.web_search_tool import web_search

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"


class WebVerdict(BaseModel):
    """Structured evidence contract shared with the local verifier."""

    stance: str          # SUPPORTED | REFUTED | INSUFFICIENT
    confidence: int      # 0-100
    reasoning: str
    sources: List[str]


def web_researcher_agent(state: AgentState) -> dict:
    """
    Web verify node (Phase 4).

    Searches trusted domains via DuckDuckGo, then uses Gemini in JSON mode (schema-
    constrained) to return the same {stance, confidence, reasoning, sources} contract
    the local verifier emits, so both feed the adjudicator uniformly.
    """
    claim = state.get("extracted_claim", "")

    # 1. Trusted-domain web search (restriction handled inside the tool).
    search_results = web_search.invoke({"query": claim})

    # 2. Schema-constrained verification.
    prompt = (
        "You are a senior fact-checking researcher. Verify the claim using ONLY the search "
        "results below, which are drawn from trusted Sri Lankan and international news "
        "organisations. If results are marked 'UNRESTRICTED', treat them with caution and "
        "require official corroboration. If the results do not clearly support or refute the "
        "claim, use stance INSUFFICIENT.\n\n"
        f"Claim: {claim}\n\n"
        f"Search Results:\n{search_results}\n\n"
        "Return: stance (SUPPORTED, REFUTED, or INSUFFICIENT), an integer confidence 0-100, a "
        "concise reasoning, and the list of trusted sources you relied on."
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=WebVerdict,
                temperature=0.0,
            ),
        )
        data = json.loads(response.text)
        stance = data.get("stance", "INSUFFICIENT")
        confidence = max(0, min(100, int(data.get("confidence", 0))))
        reasoning = (data.get("reasoning") or "").strip()
        sources = data.get("sources", []) or []
        findings = (
            f"Web stance: {stance} (confidence {confidence}%). {reasoning}"
            + (f" Sources: {', '.join(sources)}" if sources else "")
        )

        new_output = {
            "agent": "web_researcher",
            "findings": findings,
            "source_type": "trusted_web_search",
            "stance": stance,
            "confidence": confidence,
        }
        return {
            "agent_outputs": state.get("agent_outputs", []) + [new_output],
            "audit_trail": [
                {"node": "web_researcher", "status": "ok", "detail": findings[:160]}
            ],
        }
    except Exception as e:
        print(f"Error in web_researcher_agent: {e}")
        findings = f"Failed to perform web research: {e}"
        return {
            "agent_outputs": state.get("agent_outputs", [])
            + [
                {
                    "agent": "web_researcher",
                    "findings": findings,
                    "source_type": "error",
                    "stance": "INSUFFICIENT",
                    "confidence": 0,
                }
            ],
            "audit_trail": [
                {"node": "web_researcher", "status": "error", "detail": findings[:160]}
            ],
        }