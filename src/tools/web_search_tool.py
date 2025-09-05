### Search

from langchain_tavily import TavilySearch
from env_utils import TAVILY_API_KEY

web_search = TavilySearch(k=3, api_key=TAVILY_API_KEY)

if __name__ == "__main__":
    res = web_search.run("半导体优势是什么")
    print(res)