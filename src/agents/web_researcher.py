import google.generativeai as genai
from src.config import GEMINI_API_KEY
from src.state import AgentState
from src.tools.web_search_tool import web_search

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

def web_researcher_agent(state: AgentState) -> dict:
    """
    Uses Gemini and DuckDuckGo to verify general claims.
    """
    claim = state.get("extracted_claim", "")
    
    # 1. Perform web search
    search_results = web_search.invoke({"query": claim})
    
    # 2. Use Gemini to summarize and verify
    prompt = f"""
    You are a Fact-Checking Researcher. Your task is to verify the following claim using the provided search results.
    
    Claim: {claim}
    
    Search Results:
    {search_results}
    
    Tasks:
    1. Summarize the key facts found in the search results that are relevant to the claim.
    2. Identify if the claim is supported, refuted, or if there is insufficient evidence.
    3. Cite sources or general consensus where possible.
    
    Provide a structured summary of your findings.
    """
    
    try:
        response = model.generate_content(prompt)
        verification_summary = response.text.strip()
        
        # 3. Update agent_outputs
        current_outputs = state.get("agent_outputs", [])
        new_output = {
            "agent": "web_researcher",
            "findings": verification_summary,
            "source_type": "web_search"
        }
        
        return {
            "agent_outputs": current_outputs + [new_output]
        }
    except Exception as e:
        print(f"Error in web_researcher_agent: {e}")
        return {
            "agent_outputs": state.get("agent_outputs", []) + [{
                "agent": "web_researcher",
                "findings": f"Failed to perform research: {e}",
                "source_type": "error"
            }]
        }
