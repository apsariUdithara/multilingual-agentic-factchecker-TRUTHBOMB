from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

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
    Returns a summary of the top results.
    """
    try:
        search = get_search_tool()
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error during web search: {e}"
