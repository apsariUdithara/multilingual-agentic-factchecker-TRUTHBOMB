"""
Translator (Phase 5) -- best-effort EN -> SI + TA.

Takes the adjudicated verdict + English rationale and produces a short trilingual
summary via Gemini in JSON mode. Best-effort by design: on any failure it degrades
to an English-only summary and records that in the audit trail, so translation
never blocks the verdict. It merges `summaries` into the existing `final_verdict`.
"""

import json

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.config import GEMINI_API_KEY
from src.state import AgentState

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"


class Translations(BaseModel):
    english: str
    sinhala: str
    tamil: str


def _translate(verdict: str, rationale: str) -> dict:
    prompt = (
        "Summarise the following fact-check outcome in exactly two sentences, in three "
        "languages. Keep it factual and neutral.\n\n"
        f"Verdict: {verdict}\n"
        f"Rationale: {rationale}\n\n"
        "Provide the summary in English, Sinhala (සිංහල), and Tamil (தமிழ்)."
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Translations,
            temperature=0.0,
        ),
    )
    data = json.loads(response.text)
    return {
        "english": (data.get("english") or "").strip(),
        "sinhala": (data.get("sinhala") or "").strip(),
        "tamil": (data.get("tamil") or "").strip(),
    }


def translator_agent(state: AgentState) -> dict:
    """Attach trilingual summaries to the final verdict."""
    final_verdict = dict(state.get("final_verdict", {}))  # copy before mutating
    verdict = final_verdict.get("verdict", "Unverified")
    rationale = final_verdict.get("rationale", "")

    try:
        summaries = _translate(verdict, rationale)
        status, detail = "ok", "Translated verdict to EN/SI/TA."
    except Exception as e:
        print(f"Error in translator_agent: {e}")
        summaries = {
            "english": rationale or "Verdict summary unavailable.",
            "sinhala": "",
            "tamil": "",
        }
        status, detail = "error", f"Translation failed ({e}); English-only."

    final_verdict["summaries"] = summaries

    return {
        "final_verdict": final_verdict,
        "audit_trail": [{"node": "translate", "status": status, "detail": detail}],
    }