import sys
import os


# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tools.retrieval_tools import retrieve
from src.tools.web_search_tool import web_search
from langgraph.graph import StateGraph, START, END
from src.agent.node import generate, grade_documents, decide_to_generate, transform_query, direct_answer
from src.agent.conditional import grade_hallucination_and_answer, route_question
from src.agent.state import GraphState
# from langgraph.checkpoint.memory import MemorySaver
workflow = StateGraph(GraphState)

# Define the nodes we will cycle between
workflow.add_node("retrieve", retrieve)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("direct_answer", direct_answer)


# 先从路由节点开始 判断是web搜索还是进入数据库
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vector": "retrieve",
        "web": "web_search",
        "direct_answer": "direct_answer",
    }
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("direct_answer", END)
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate_answer": "generate",
        "rewrite_question": "transform_query",
    }
)
workflow.add_conditional_edges(
    "generate",
    grade_hallucination_and_answer,
    {
        "useful": END,
        "not useful": "transform_query",
        "not supported": "generate",
    }
)


# 检查点让状态图可以持久化其状态
# 这是整个状态图的完整内存
# memory = MemorySaver()

# 编译状态图，配置检查点为memory, 配置中断点 如果用langgraph dev就不用传入记忆
# graph = workflow.compile(checkpointer=memory)
graph = workflow.compile()
# 画图 - 输出 Mermaid 代码到控制台
# print("LangGraph 流程图:")
# print(graph.get_graph().draw_mermaid())