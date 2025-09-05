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
    search_results = tavily_search.invoke(question)
    
    # 解析Tavily搜索结果，提取实际内容
    documents = []
    if isinstance(search_results, list):
        # 如果返回的是结果列表
        for result in search_results:
            if isinstance(result, dict):
                # 提取标题和内容
                content = ""
                if "content" in result:
                    content = result["content"]
                elif "snippet" in result:
                    content = result["snippet"]
                elif "summary" in result:
                    content = result["summary"]
                
                if content:
                    documents.append(content)
            elif isinstance(result, str):
                documents.append(result)
    elif isinstance(search_results, dict):
        # 如果返回的是字典格式
        if "results" in search_results:
            for result in search_results["results"]:
                content = ""
                if isinstance(result, dict):
                    if "content" in result:
                        content = result["content"]
                    elif "snippet" in result:
                        content = result["snippet"]
                    elif "summary" in result:
                        content = result["summary"]
                    
                    if content:
                        documents.append(content)
        elif "answer" in search_results:
            # 如果有直接答案
            documents.append(search_results["answer"])
    
    # 如果没有找到任何内容，使用原始结果
    if not documents and search_results:
        documents = [str(search_results)]
    
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    res = web_search({"question": "半导体优势是什么"})
    print(res)