import sys
import os
from typing import List

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langgraph.graph import MessagesState
from llm_utils import llm
from src.tools.retrieval_tools import retrieval_tool
from langchain_core.messages import HumanMessage, convert_to_messages, BaseMessage
from src.agent.prompt import REWRITE_PROMPT, GENERATE_PROMPT
from utils.log_utils import log
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

model_with_tools = llm.bind_tools([retrieval_tool])
# 获取最后一个HumanMessage
def get_last_human_message(messages: List[BaseMessage]) -> HumanMessage:
    """Get the last HumanMessage from a list of messages"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    raise ValueError("No HumanMessage found in messages")

def generate_query_or_respond(state: MessagesState) -> MessagesState:
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    log.info("*****Start generate a query or respond using the model*****")
    res = (
        model_with_tools.invoke([state['messages'][-1]])          # 我们把第一次用户的HumanMessage传给模型
    )
    return {
        'messages': [res]
    }

# 写法1 用.format 格式化字符串的写法
def rewrite_question(state: MessagesState) -> MessagesState:
    """Rewrite the question using the model"""
    log.info("*****Start rewrite the question using the model*****")    
    messages = state['messages']
    question = get_last_human_message(messages).content

    # 使用字符串格式化替换占位符
    formatted_prompt = REWRITE_PROMPT.format(question=question)
    
    msg = HumanMessage(content=formatted_prompt)
    res = llm.invoke([msg])  # 注意这里传入的是消息列表
    
    return {
        'messages': [res]
    }
# 写法2 用chain的写法
# def rewrite_question(state: MessagesState) -> MessagesState:
#     """Rewrite the question using the model"""
#     messages = state['messages']
#     question = get_last_human_message(messages).content
    
#     # 创建PromptTemplate并使用管道
#     prompt_template = PromptTemplate(template=REWRITE_PROMPT, input_variables=["question"])
#     chain_rewrite = (
#         prompt_template
#         | llm
#         | StrOutputParser()
#     )
    
#     response = chain_rewrite.invoke({"question": question})       # 这里拿到的直接就是文本
#     ai_msg = AIMessage(content=response)   
#     return {
#         'messages': [ai_msg]
#     }
# 写法1 用.format 格式化字符串的写法
def generate_answer(state: MessagesState) -> MessagesState:
    """Generate an answer using the model"""
    log.info("*****Start generate an answer using the model*****")
    messages = state['messages']
    question = get_last_human_message(messages).content      # 第一个节点的content就是用户问题
    context = messages[-1].content      # 上一个节点就是retrieval检索节点，他的content就是检索结果
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    res = model_with_tools.invoke([HumanMessage(content=prompt)])
    return {
        # 'messages': messages + [res]       # 这样会保存所有的消息，包括原始问题和检索结果
        'messages': [res]
    }
# 写法2 用chain的写法
# def generate_answer(state: MessagesState) -> MessagesState:
#     """Generate an answer using the model"""
#     log.info("*****Start generate an answer using the model*****")
#     messages = state['messages']

#     question = get_last_human_message(messages).content      # 第一个节点的content就是用户问题
#     context = messages[-1].content      # 上一个节点就是retrieval检索节点，他的content就是检索结果

#     generate_prompt = PromptTemplate(template=GENERATE_PROMPT, input_variables=["question", "context"])
#     chain_generate = (
#         generate_prompt
#         | llm
#         | StrOutputParser()
#     )
#     res = chain_generate.invoke({"question": question, "context": context})
#     ai_msg = AIMessage(content=res)
#     return {
#         'messages': [ai_msg]
#     }

if __name__ == "__main__":
    # input = {"messages": [HumanMessage(content="半导体优势是什么")]}
    # res = generate_query_or_respond(input)
    # print(res)
    # input = {"messages": [HumanMessage(content="半导体优势是什么")]}
    # res = rewrite_question(input)
    # res['messages'][-1].pretty_print()
    input = {
        "messages": convert_to_messages(
            [
                {
                    "role": "user",
                    "content": "半导体优势是什么"
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "1",
                            "name": "retrieval_tool",
                            "args": {
                                "query": "半导体优势是什么"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "content": "半导体优势是能够提高生产效率和产品质量，降低成本，提高竞争力。",
                    "tool_call_id": "1",
                }
            ]
        )
    }
    res = generate_answer(input)
    print(res)