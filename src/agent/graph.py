import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from src.agent.node import generate_query_or_respond, rewrite_question, generate_answer
from src.tools.retrieval_tools import retrieval_tool
from src.agent.conditional import grade_documents
from langgraph.checkpoint.memory import MemorySaver
workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("retrieval", ToolNode([retrieval_tool]))


workflow.add_edge(START, "generate_query_or_respond")
# 第一次conditional_edge 如果 generate_query_or_respond 返回了 tool_calls ，则调用 retriever_tool 以获取上下文 否则直接响应用户
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        "tools": "retrieval",
        END: END
    }
)

# 第二次conditional_edge 对检索到的文档内容进行与问题相关性的评分（ grade_documents ），并路由至下一步：
# 若不相关，使用 rewrite_question 重写问题，然后再次调用 若相关，继续执行 generate_answer ，并使用检索到的文档上下文通过 ToolMessage 生成最终响应
workflow.add_conditional_edges(
    "retrieval",
    grade_documents,
    {
        "accept": "generate_answer",
        "ignore": "rewrite_question",
    }
)

workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")



# 检查点让状态图可以持久化其状态
# 这是整个状态图的完整内存
memory = MemorySaver()

# 编译状态图，配置检查点为memory, 配置中断点
graph = workflow.compile(checkpointer=memory)

# 画图 - 输出 Mermaid 代码到控制台
# print("LangGraph 流程图:")
# print(graph.get_graph().draw_mermaid())