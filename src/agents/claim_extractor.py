from src.state import AgentState
from src.models.extractor import get_extractor


def claim_extractor_agent(state: AgentState) -> dict:
    """
    Dual-Debater Claim Extractor Agent Node.
    Runs Debater A (Fine-Tuned GGUF Model) and Debater B (Cloud Gemini LLM) in parallel
    to compare extraction perspectives, establish initial consensus status, and route execution.
    """
    raw_input = state.get("raw_input", "")
    
    # --- DEBATER A: Fine-Tuned Local GGUF Model ---
    extracted_claim_a = ""
    try:
        import src.main as main_module
        if main_module.extractor_model is not None:
            extracted_claim_a = main_module.extractor_model.extract_claim(raw_input)
    except Exception as e:
        print(f"Debater A (GGUF) error: {e}")
        
    if not extracted_claim_a:
        try:
            from src.models.extractor import LinguisticExtractor
            local_extractor = LinguisticExtractor()
            extracted_claim_a = local_extractor.extract_claim(raw_input)
        except Exception as e:
            print(f"Debater A lazy fallback failed: {e}")

    # --- DEBATER B & ROUTER: Cloud Gemini API ---
    prompt = f"""
    You are Debater B (Cloud Gemini LLM) & Fact-Checking Router.
    Analyze the following raw social media post and perform two tasks:

    Tasks:
    1. Extract the singular verifiable core claim in clean, professional English (Debater B perspective).
    2. Identify the primary language (English, Sinhala, Tamil, or Singlish/Tanglish).
    3. Categorize the research routing destination:
       - 'local_researcher' if local Sri Lankan news / internal OSINT database topic.
       - 'web_researcher' if general / international news or requires real-time web search.

    Raw Post:
    {raw_input}

    Return response EXACTLY as JSON:
    {{
        "extracted_claim_b": "str",
        "language": "str",
        "routing_destination": "str"
    }}
    """
    
    extracted_claim_b = ""
    language = "English"
    routing_destination = "web_researcher"
    
    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace('```json', '').replace('```', '')
        result = json.loads(cleaned_response)
        
        extracted_claim_b = result.get("extracted_claim_b", "")
        language = result.get("language", "English")
        routing_destination = result.get("routing_destination", "web_researcher")
    except Exception as e:
        print(f"Debater B (Gemini) error: {e}")
        extracted_claim_b = raw_input

    # Normalize fallback values
    if not extracted_claim_a:
        extracted_claim_a = extracted_claim_b if extracted_claim_b else raw_input
    if not extracted_claim_b:
        extracted_claim_b = extracted_claim_a

    # Determine initial consensus (exact/near match vs divergence)
    claim_a_clean = extracted_claim_a.lower().strip()
    claim_b_clean = extracted_claim_b.lower().strip()
    
    if claim_a_clean == claim_b_clean or (len(claim_a_clean) > 10 and claim_a_clean[:20] == claim_b_clean[:20]):
        consensus_status = "AGREEMENT"
    else:
        consensus_status = "DISAGREEMENT"

    # Select primary claim for RAG retrieval: prefer Debater B if Debater A added unnecessary speculative clauses
    if extracted_claim_b and (len(extracted_claim_b) < len(extracted_claim_a) or raw_input.lower().strip() in extracted_claim_b.lower()):
        primary_claim = extracted_claim_b
    elif extracted_claim_a:
        primary_claim = extracted_claim_a
    else:
        primary_claim = raw_input

    return {
        "language": language,
        "extracted_claim": primary_claim,
        "extracted_claim_a": extracted_claim_a,
        "extracted_claim_b": extracted_claim_b,
        "consensus_status": consensus_status,
        "routing_destination": routing_destination
    }
