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
    Search is restricted to trusted domains to ensure data credibility.
    """
    claim = state.get("extracted_claim", "")
    
    # 1. Perform web search (Restricted to trusted sites in tool)
    search_results = web_search.invoke({"query": claim})
    
    # 2. Use Gemini to summarize and verify
    prompt = f"""
    You are a Senior Fact-Checking Researcher. Your task is to verify the following claim.
    
    CRITICAL: The search results provided are specifically retrieved from TRUSTED news organizations 
    (e.g., Ada Derana, NewsFirst, Daily Mirror, Official Govt Portal) to ensure high credibility.
    
    Claim: {claim}
    
    Search Results:
    {search_results}
    
    Tasks:
    1. Summarize the findings from these authoritative sources.
    2. Explicitly state which trusted site provided the information.
    3. Determine if the claim is SUPPORTED, REFUTED, or UNVERIFIED based on these specific results.
    4. If the results are marked as 'UNRESTRICTED', treat them with extreme caution and look for official corroboration.
    
    Provide a professional, source-cited summary.
    """
    
    try:
        response = model.generate_content(prompt)
        verification_summary = response.text.strip()
        
        # 3. Update agent_outputs
        current_outputs = state.get("agent_outputs", [])
        new_output = {
            "agent": "web_researcher",
            "findings": verification_summary,
            "source_type": "trusted_web_search"
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
