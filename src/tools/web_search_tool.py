from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool

search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """
    Searches the web for general information and latest news using DuckDuckGo.
    Returns a summary of the top results.
    """
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"Error during web search: {e}"
