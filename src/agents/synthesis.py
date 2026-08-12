import google.generativeai as genai
import json
from src.config import GEMINI_API_KEY
from src.state import AgentState

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

def synthesis_agent(state: AgentState) -> dict:
    """
    Chief Fact-Checking Arbitrator & Judge Node.
    Compares Debater A (Fine-Tuned GGUF) and Debater B (Cloud Gemini LLM),
    evaluates both perspectives against ground-truth evidence, resolves divergence,
    and produces a structured final verdict.
    """
    claim_a = state.get("extracted_claim_a", state.get("extracted_claim", ""))
    claim_b = state.get("extracted_claim_b", state.get("extracted_claim", ""))
    initial_consensus = state.get("consensus_status", "AGREEMENT")
    agent_outputs = state.get("agent_outputs", [])
    
    evidence_str = ""
    for output in agent_outputs:
        evidence_str += f"\n--- Researcher Agent: {output['agent']} ({output.get('source_type', 'unknown')}) ---\nFindings: {output['findings']}\n"
    
    prompt = f"""
    You are the Chief Fact-Checking Arbitrator & Judge for a Sri Lankan Social Media Fact-Checking Platform.
    Two independent AI debaters have extracted the claim from the raw user post:

    - Debater A (Fine-Tuned Local GGUF Model): "{claim_a}"
    - Debater B (Cloud Gemini LLM): "{claim_b}"
    - Initial Consensus Status: {initial_consensus}

    Retrieved Research Evidence:
    {evidence_str}

    Tasks:
    1. Compare the claims extracted by Debater A and Debater B.
    2. Determine the final consensus status: 'AGREEMENT' or 'DISAGREEMENT'.
    3. Evaluate both claims against the Retrieved Research Evidence:
       - CRITICAL RULE: If a social media post or headline states a news event (e.g. 'X further remanded') and the retrieved official news reports confirm the core event (e.g. 'Sugeeshwara Bandara remanded until August 25'), the verdict MUST be 'True'.
       - Do NOT declare a claim 'False' or nitpick minor phrasing details (such as whether 'further investigation' was mentioned) if the central headline news event is factually true and confirmed by official news feeds!
       - If evidence directly contradicts the core news event, verdict is 'False'.
       - If evidence does not mention the event at all, verdict MUST be 'Unverified'.
    4. Determine final verdict: True, False, Misleading, or Unverified.
    5. Assign a confidence score (0% - 100%).
    6. Provide a detailed rationale in English explaining the verdict and arbitration.
    7. Provide a 2-sentence summary of the verdict in:
       - English
       - Sinhala (සිංහල)
       - Tamil (தமிழ்)

    Return result EXACTLY as a JSON object:
    {{
        "verdict": "str",
        "confidence_score": "str",
        "consensus_status": "AGREEMENT|DISAGREEMENT",
        "debater_a_claim": "str",
        "debater_b_claim": "str",
        "rationale": "str",
        "summaries": {{
            "english": "str",
            "sinhala": "str",
            "tamil": "str"
        }}
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        final_verdict = json.loads(cleaned_response)
        
        # Ensure debater claims are present in final payload
        if "debater_a_claim" not in final_verdict:
            final_verdict["debater_a_claim"] = claim_a
        if "debater_b_claim" not in final_verdict:
            final_verdict["debater_b_claim"] = claim_b
        if "consensus_status" not in final_verdict:
            final_verdict["consensus_status"] = initial_consensus

        return {
            "final_verdict": final_verdict
        }
    except Exception as e:
        print(f"Error in synthesis_agent: {e}")
        return {
            "final_verdict": {
                "verdict": "Unverified",
                "confidence_score": "0%",
                "consensus_status": initial_consensus,
                "debater_a_claim": claim_a,
                "debater_b_claim": claim_b,
                "rationale": f"Error during arbitration: {e}",
                "summaries": {
                    "english": "Could not verify the claim due to a system error.",
                    "sinhala": "පද්ධති දෝෂයක් හේතුවෙන් මෙම පකාශය තහවුරු කිරීමට නොහැකි විය.",
                    "tamil": "கணினி பிழை காரணமாக உரிமைகோரலை சரிபார்க்க முடியவில்லை."
                }
            }
        }
