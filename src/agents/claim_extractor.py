from src.state import AgentState
from src.models.extractor import get_extractor


def claim_extractor_agent(state: AgentState) -> dict:
    """
    Analyzes the raw input to extract a verifiable claim (using the fine-tuned local GGUF model
    via LinguisticExtractor), identify the language, and determine routing.
    """
    raw_input = state.get("raw_input", "")
    
    # 1. Reuse the already loaded local GGUF extractor model from src.main
    extracted_claim = ""
    try:
        import src.main as main_module
        if main_module.extractor_model is not None:
            extracted_claim = main_module.extractor_model.extract_claim(raw_input)
    except Exception as e:
        print(f"Error calling local extractor model, falling back to lazy initialization: {e}")
        
    # If the GGUF instance was not available, initialize it lazily
    if not extracted_claim:
        try:
            from src.models.extractor import LinguisticExtractor
            local_extractor = LinguisticExtractor()
            extracted_claim = local_extractor.extract_claim(raw_input)
        except Exception as e:
            print(f"Lazy local extractor failed: {e}")
            
    # 2. Use Gemini to perform lightweight routing classification (language & destination)
    # based on the raw input and the GGUF-extracted English claim.
    prompt = f"""
    You are a Fact-Checking Routing Assistant.
    Based on the raw social media post and the extracted English claim, complete two tasks:
    1. Identify the primary language of the raw post (English, Sinhala, Tamil, or Singlish/Tanglish).
    2. Categorize the research routing destination:
       - If the claim is about local events, government announcements, or specific Sri Lankan news likely found in our internal database, route it to 'local_researcher'.
       - If the claim is about international news, general knowledge, or requires real-time broad internet searching, route it to 'web_researcher'.

    Raw Post: {raw_input}
    Extracted Claim (English): {extracted_claim}

    Return the result EXACTLY as a JSON object with the following keys:
    {{
        "language": "str",
        "routing_destination": "str"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(cleaned_response)
        
        return {
            "language": result.get("language", "English"),
            "extracted_claim": extracted_claim if extracted_claim else raw_input,
            "routing_destination": result.get("routing_destination", "web_researcher")
        }
    except Exception as e:
        print(f"Error in claim_extractor_agent classification: {e}")
        # Default fallback
        return {
            "language": "English",
            "extracted_claim": extracted_claim if extracted_claim else raw_input,
            "routing_destination": "web_researcher"
        }
