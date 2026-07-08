import json

from google import genai

from src.config import GEMINI_API_KEY
from src.state import AgentState

# New unified SDK: the Client IS the callable surface (no GenerativeModel object).
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"


def synthesis_agent(state: AgentState) -> dict:
    """
    Combines evidence from all agents and provides a final verdict in multiple languages.
    """
    claim = state.get("extracted_claim", "")
    agent_outputs = state.get("agent_outputs", [])

    evidence_str = ""
    for output in agent_outputs:
        evidence_str += f"\n--- Agent: {output['agent']} ---\nFindings: {output['findings']}\n"

    prompt = f"""
    You are a Final Verdict Specialist for a Sri Lankan Fact-Checking System.
    Based on the evidence gathered by researchers, provide a final verdict on the claim.
    
    Original Claim: {claim}
    
    Evidence Gathered:
    {evidence_str}
    
    Tasks:
    1. Determine the final verdict: True, False, Misleading, or Unverified.
    2. Assign a confidence score (0% - 100%).
    3. Provide a detailed rationale in English.
    4. Provide a 2-sentence summary of the verdict in:
       - English
       - Sinhala (සිංහල)
       - Tamil (தமிழ்)

    Return the result EXACTLY as a JSON object with this structure:
    {{
        "verdict": "str",
        "confidence_score": "str",
        "rationale": "str",
        "summaries": {{
            "english": "str",
            "sinhala": "str",
            "tamil": "str"
        }}
    }}
    """

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        final_verdict = json.loads(cleaned_response)

        return {"final_verdict": final_verdict}
    except Exception as e:
        print(f"Error in synthesis_agent: {e}")
        return {
            "final_verdict": {
                "verdict": "Unverified",
                "confidence_score": "0%",
                "rationale": f"Error during synthesis: {e}",
                "summaries": {
                    "english": "Could not verify the claim due to a system error.",
                    "sinhala": "පද්ධති දෝෂයක් හේතුවෙන් මෙම පකාශය තහවුරු කිරීමට නොහැකි විය.",
                    "tamil": "கணினி பிழை காரணமாக உரிமைகோரலை சரிபார்க்க முடியவில்லை.",
                },
            }
        }