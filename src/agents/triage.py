import google.generativeai as genai
import json
from src.config import GEMINI_API_KEY
from src.state import AgentState

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def triage_agent(state: AgentState) -> dict:
    """
    Parses the raw input, extracts the claim, identifies the language, 
    and determines the routing destination.
    """
    raw_input = state.get("raw_input", "")
    
    # Check if raw_input is a path to an image or just text
    # For now, we assume it's text. If it's an image path, 
    # we would need to load the image.
    
    prompt = f"""
    You are a Triage Specialist for a Sri Lankan Social Media Fact-Checking System.
    Analyze the following input:
    ---
    {raw_input}
    ---
    
    Tasks:
    1. Identify the primary language (English, Sinhala, Tamil, or Singlish/Tanglish).
    2. Extract the singular verifiable core claim. Strip out emotional language, opinions, and insults.
    3. Categorize the domain.
       - If the claim relates to Sri Lankan traffic fines, vehicle registrations, road rules, driving laws, or highway regulations, route it to 'traffic_expert'.
       - Otherwise, route it to 'web_researcher'.

    Return the result EXACTLY as a JSON object with the following keys:
    {{
        "language": "str",
        "extracted_claim": "str",
        "routing_destination": "str"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(cleaned_response)
        
        return {
            "language": result.get("language", "English"),
            "extracted_claim": result.get("extracted_claim", ""),
            "routing_destination": result.get("routing_destination", "web_researcher")
        }
    except Exception as e:
        print(f"Error in triage_agent: {e}")
        # Fallback values
        return {
            "language": "English",
            "extracted_claim": raw_input,
            "routing_destination": "web_researcher"
        }
