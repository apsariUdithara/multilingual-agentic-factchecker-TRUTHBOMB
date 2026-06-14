from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from src.config import TRUSTED_NEWS_DOMAINS

_search_tool = None

def get_search_tool():
    global _search_tool
    if _search_tool is None:
        _search_tool = DuckDuckGoSearchRun()
    return _search_tool

@tool
def web_search(query: str) -> str:
    """
    Searches the web for general information and latest news using DuckDuckGo.
    Specifically restricted to trusted Sri Lankan and international news domains.
    """
    try:
        # Construct a restricted query using 'site:' operators
        # Example: "claim site:adaderana.lk OR site:newsfirst.lk"
        domain_filter = " OR ".join([f"site:{domain}" for domain in TRUSTED_NEWS_DOMAINS])
        restricted_query = f"{query} ({domain_filter})"
        
        search = get_search_tool()
        results = search.run(restricted_query)
        return results
    except Exception as e:
        # Fallback to general search if the restricted one fails, 
        # but mark it so the agent knows.
        try:
            search = get_search_tool()
            results = search.run(query)
            return f"UNRESTRICTED RESULTS (Trusted search failed): {results}"
        except Exception as e2:
            return f"Error during web search: {e2}"
