### Search

from langchain_tavily import TavilySearch
from env_utils import TAVILY_API_KEY

tavily_search = TavilySearch(k=3, api_key=TAVILY_API_KEY)

def web_search(state):
    """
    Web search based on the query

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains web search results
    """
    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    documents = tavily_search.invoke(question)
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    res = web_search({"question": "半导体优势是什么"})
    print(res)