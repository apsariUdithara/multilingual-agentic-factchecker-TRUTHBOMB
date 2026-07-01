import json

from google import genai

from src.config import GEMINI_API_KEY
from src.state import AgentState

# New unified SDK: the Client IS the callable surface (no GenerativeModel object).
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"


def claim_extractor_agent(state: AgentState) -> dict:
    """
    Analyzes the raw input to extract a verifiable claim, identify the language,
    and determine if it should be researched locally or via web search.
    """
    raw_input = state.get("raw_input", "")

    prompt = f"""
    You are a Claim Extraction Specialist for a Sri Lankan Social Media Fact-Checking System.
    Analyze the following input:
    ---
    {raw_input}
    ---
    
    Tasks:
    1. Identify the primary language (English, Sinhala, Tamil, or Singlish/Tanglish).
    2. Extract the singular verifiable core claim. Normalize it into professional English.
       - Strip out emotional language, opinions, and social media noise.
    3. Categorize the research requirement:
       - If the claim is about local events, government announcements, or specific Sri Lankan news likely found in our internal database, route it to 'local_researcher'.
       - If the claim is about international news, general knowledge, or requires real-time broad internet searching, route it to 'web_researcher'.

    Return the result EXACTLY as a JSON object with the following keys:
    {{
        "language": "str",
        "extracted_claim": "str",
        "routing_destination": "str"
    }}
    """

    try:
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned_response)

        return {
            "language": result.get("language", "English"),
            "extracted_claim": result.get("extracted_claim", ""),
            "routing_destination": result.get("routing_destination", "web_researcher"),
        }
    except Exception as e:
        print(f"Error in claim_extractor_agent: {e}")
        return {
            "language": "English",
            "extracted_claim": raw_input,
            "routing_destination": "web_researcher",
        }