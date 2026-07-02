"""
Adjudicator (Phase 5) -- rules-first, Gemini only when unresolved.

Maps the structured stances emitted by the verifiers (SUPPORTED / REFUTED /
INSUFFICIENT + confidence) onto the canonical verdict vocabulary. Deterministic
for clear-cut and clearly-weak evidence; delegates to Gemini only for genuinely
borderline single stances or conflicting stances (the only paths that can yield
a nuanced "Misleading" verdict).

The chosen path is recorded as `resolution_method` ("rules" | "gemini") in the
verdict and the audit trail, so the decision is always traceable.
"""

import json

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.config import (
    GEMINI_API_KEY,
    VERDICT_TRUE,
    VERDICT_FALSE,
    VERDICT_MISLEADING,
    VERDICT_UNVERIFIED,
    ADJUDICATION_HIGH_CONFIDENCE,
    ADJUDICATION_MIN_CONFIDENCE,
)
from src.state import AgentState

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

_ALLOWED_VERDICTS = {VERDICT_TRUE, VERDICT_FALSE, VERDICT_MISLEADING, VERDICT_UNVERIFIED}


class Adjudication(BaseModel):
    verdict: str          # True | False | Misleading | Unverified
    confidence_score: int  # 0-100
    rationale: str


def _gemini_adjudicate(claim: str, evidence: str) -> dict:
    """Resolve borderline/conflicting evidence with Gemini (schema-constrained)."""
    prompt = (
        "You are the final adjudicator in a Sri Lankan fact-checking system. The automated "
        "verifiers were inconclusive or in conflict. Weigh the evidence and decide.\n\n"
        f"Claim: {claim}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Choose exactly one verdict: True, False, Misleading (partially true or deceptively "
        "framed), or Unverified. Give an integer confidence 0-100 and a concise rationale."
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Adjudication,
            temperature=0.0,
        ),
    )
    data = json.loads(response.text)
    verdict = data.get("verdict", VERDICT_UNVERIFIED)
    if verdict not in _ALLOWED_VERDICTS:
        verdict = VERDICT_UNVERIFIED
    confidence = max(0, min(100, int(data.get("confidence_score", 0))))
    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": (data.get("rationale") or "").strip(),
    }


def adjudicator_agent(state: AgentState) -> dict:
    """Produce the final verdict from the gathered stances."""
    claim = state.get("extracted_claim", "")
    outputs = state.get("agent_outputs", [])

    supported = [int(o.get("confidence", 0)) for o in outputs if o.get("stance") == "SUPPORTED"]
    refuted = [int(o.get("confidence", 0)) for o in outputs if o.get("stance") == "REFUTED"]
    evidence = "\n".join(o.get("findings", "") for o in outputs).strip() or "No evidence was gathered."

    method = "rules"

    if supported and refuted:
        # Conflict -> unresolved -> Gemini.
        try:
            result = _gemini_adjudicate(claim, evidence)
            verdict, confidence, rationale, method = (
                result["verdict"], result["confidence"], result["rationale"], "gemini",
            )
        except Exception as e:
            print(f"Error in Gemini adjudication (conflict): {e}")
            verdict, confidence, rationale = VERDICT_UNVERIFIED, 0, f"Conflicting evidence; adjudication failed ({e})."
    elif supported or refuted:
        is_supported = bool(supported)
        confidence = max(supported) if is_supported else max(refuted)
        if confidence >= ADJUDICATION_HIGH_CONFIDENCE:
            verdict = VERDICT_TRUE if is_supported else VERDICT_FALSE
            rationale = evidence
        elif confidence >= ADJUDICATION_MIN_CONFIDENCE:
            # Borderline -> Gemini (may return Misleading).
            try:
                result = _gemini_adjudicate(claim, evidence)
                verdict, confidence, rationale, method = (
                    result["verdict"], result["confidence"], result["rationale"], "gemini",
                )
            except Exception as e:
                print(f"Error in Gemini adjudication (borderline): {e}")
                verdict = VERDICT_TRUE if is_supported else VERDICT_FALSE
                rationale = evidence
        else:
            verdict, rationale = VERDICT_UNVERIFIED, evidence
    else:
        # No usable stance (all INSUFFICIENT or none).
        verdict, confidence, rationale = VERDICT_UNVERIFIED, 0, evidence

    final_verdict = {
        "verdict": verdict,
        "confidence_score": f"{confidence}%",
        "rationale": rationale.strip(),
        "resolution_method": method,
    }

    return {
        "final_verdict": final_verdict,
        "audit_trail": [
            {
                "node": "adjudicate",
                "status": "ok",
                "detail": f"{verdict} via {method} (confidence {confidence}%).",
            }
        ],
    }