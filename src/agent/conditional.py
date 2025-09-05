import sys
import os

from langchain_core.prompts import ChatPromptTemplate
# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.agent.prompt import ROUTER_FOR_QUERY_ANALYSIS_PROMPT, RETRIEVAL_GRADE_PROMPT, HALLUCINATION_GRADE_PROMPT, ANSWER_RELEVANCE_GRADE_PROMPT
from pydantic import BaseModel, Field
from llm_utils import llm
from typing import Literal
from utils.log_utils import log
from src.agent.state import GraphState
# from langchain_core.messages import HumanMessage
# from src.agent.node import get_last_human_message


# *******************定义第一次Query到web_search或者vector_database的路由函数*******************
# 定义第一次Query到web_search或者vector_database的路由函数
class RouteQueryAnalysis(BaseModel):
    """Route query analysis to the most relevant datasource"""
    datasource: Literal["vector_database", "web_search", "direct_answer"] = Field(
        ...,
        description="Given a user question choose to route it to a vector database or a web search or direct answer"
    )
# Query_route提示词写成Runnable
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ROUTER_FOR_QUERY_ANALYSIS_PROMPT),
        ("human", "{question}")
    ]
)

question_router = route_prompt | llm.with_structured_output(RouteQueryAnalysis)

# 第一个路由函数，判断检索到的文档是否与问题相关 如果相关返回"vector"，否则返回"web"
def route_question(state: GraphState) -> str:
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    log.info("*****Start route the question to web search or RAG or direct answer*****")
    question = state['question']
    source = question_router.invoke({"question": question})
    if source.datasource == "vector_database":
        print("Route to vector database")
        return "vector"
    elif source.datasource == "web_search":
        print("Route to web search")
        return "web"
    else:   
        print("Route to direct answer")
        return "direct_answer"
    
# *******************2. 检索分级器 定义检索分级器执行检索后，评估结果。虽然最初根据查询决定使用 RAG，但检索到的文档可能并不令人满意。评估检索到的文档是否与查询足够相关。*******************
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: Literal["yes", "no"] = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
retrieval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RETRIEVAL_GRADE_PROMPT),
        ("human", "User question: {question}\n\n Retrieved documents: {documents}")
    ]
)
retrieval_grade = retrieval_prompt | llm.with_structured_output(GradeDocuments)

# *******************3. 幻觉评分器，将 LLM 的输出与检索到的事实进行对比，验证其是否产生任何幻觉内容。*******************
class HallucinationGrade(BaseModel):
    """Binary score for hallucination check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_GRADE_PROMPT),
        ("human", "Set of facts: {documents}\n\n LLM Generated Answer: {generation}")
    ]
)
hallucination_grade = hallucination_prompt | llm.with_structured_output(HallucinationGrade)
# *******************4. 答案评分器 评估答案是否解决了相关问题*******************
# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: Literal["yes", "no"] = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_RELEVANCE_GRADE_PROMPT),
        ("human", "User question: {question}\n\n LLM Generated Answer: {generation}")
    ]
)
# Chain
answer_grade = answer_prompt | llm.with_structured_output(GradeAnswer)

# 幻觉评测 + 答案评测 如果都通过的生成路由函数 整个 Adaptive RAG 流程的最终质量把关环节，负责双重验证
def grade_hallucination_and_answer(state):
    """
    Grade hallucination and answer
    """
    log.info("*****Start grade hallucination and answer*****")
    documents = state["documents"]
    generation = state["generation"]
    question = state["question"]

    hallucination_score = hallucination_grade.invoke({"documents": documents, "generation": generation})
    # 先检查幻觉 如果yes说明 LLM 生成的内容是基于/受一组检索到的 事实支持的 继续进行   答案评测
    if hallucination_score.binary_score == "yes":
        print("Hallucination check passed")
        answer_score = answer_grade.invoke({"question": question, "generation": generation})
        if answer_score.binary_score == "yes":
            print("Answer check passed")
            return "useful"
        else:
            print("Answer check failed")
            return "not useful"
    else:
        print("Hallucination check failed")
        return "not supported"
    





if __name__ == "__main__":
    res1 = question_router.invoke({"question": "hello"})
    print(res1)
    # # res2 = question_router.invoke({"question": "秦始皇是谁"})
    # # print(res2)
    # # res = answer_grade.invoke({"question": "半导体优势是什么", "answer": "半导体优势是能够提高生产效率和产品质量，降低成本，提高竞争力。"})
    # # print(res)
    # # res = retrieval_grade.invoke({"question": "半导体优势是什么", "documents": "半导体优势是能够提高生产效率和产品质量，降低成本，提高竞争力。"})
    # # print(res)
    # # res = hallucination_grade.invoke({"documents": "半导体优势是能够提高生产效率和产品质量，降低成本，提高竞争力。", "generation": "半导体优势是降低成本，提高竞争力。"})
    # # print(res)
    # res = answer_grade.invoke({"question": "半导体优势是什么", "generation": "半导体优势是降低成本，提高竞争力。"})
    # print(res)
