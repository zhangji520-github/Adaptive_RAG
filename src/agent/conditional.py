"""
Grade documents:
    添加条件边—— grade_documents ——用于判断检索到的文档是否与问题相关。我们将使用具有结构化输出模式 GradeDocuments 的模型进行文档评分。 
    grade_documents 函数将根据评分结果（ generate_answer 或 rewrite_question ）返回下一步要执行的节点名称。
    using a binary score for relevance check
"""
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agent.prompt import GRADE_PROMPT
from pydantic import BaseModel, Field
from llm_utils import llm
from langgraph.graph import MessagesState
from typing import Literal
from utils.log_utils import log
from langchain_core.messages import HumanMessage


# 定义结构化输出的state
class GradeDocuments(BaseModel):
    """Grade documents based on relevance to the question"""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )
    
# 定义路由函数，如果binary_score为yes，则返回accept，否则返回ignore
def grade_documents(state: MessagesState) -> Literal["accept", "ignore"]:
    """Grade documents based on relevance to the question"""
    log.info("*****Start grade the documents based on relevance to the question*****")
    messages = state['messages']
    question = messages[0].content
    context = messages[-1].content
    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = llm.with_structured_output(GradeDocuments).invoke([HumanMessage(content=prompt)])
    if response.binary_score == "yes":
        return "accept"
    else:
        return "ignore"
