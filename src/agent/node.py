import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_core.prompts import ChatPromptTemplate
from llm_utils import llm
from src.agent.prompt import QUESTIONING_REWRITING_PROMPT, RAG_PROMPT_TEMPLATE
from utils.log_utils import log
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.agent.conditional import retrieval_grade
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate







# *******************生成答案节点*******************
# post_processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    log.info("*****Start generate answer*****")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation - 将字符串模板转换为ChatPromptTemplate
    rag_prompt = ChatPromptTemplate.from_messages([
        ("human", RAG_PROMPT_TEMPLATE)
    ])
    rag_chain = rag_prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# *******************对通过向量数据库检索（retrieve 节点）得到的文档进行相关性评估，过滤掉与用户问题无关的内容*******************
def grade_documents(state):
    """
    Grade documents
    """
    log.info("*****Start grade documents*****")
    question = state["question"]
    documents = state["documents"]
    
    filter_doc = []
    for d in documents:
        score = retrieval_grade.invoke({"question": question, "documents": d})
        if score.binary_score == "yes":
            filter_doc.append(d)
        else:
            continue
    return {"documents": filter_doc, "question": question}

# 根据过滤情况选择是否生成/回退重写问题 路由函数
def decide_to_generate(state):
    """
    Decide to generate or rewrite question
    """
    log.info("*****Start decide to generate or rewrite question*****")
    filtered_documents = state["documents"]
    if not filtered_documents:
        print("😒Decision: ALL documents are irrelevant to the question. I think I should rewrite the question.")
        return "rewrite_question"
    else:
        print("😊Decision: Some documents are relevant to the question. I think I should generate the answer.")
        return "generate_answer"
    
# *******************问题重写节点 通过提示词在转化成Runnable对象*******************
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QUESTIONING_REWRITING_PROMPT),
        ("human", "Here is the initial question: \n\n {question}\n\n Formulate an improved question.")
    ]
)
question_rewriter = rewrite_prompt | llm | StrOutputParser()    
def transform_query(state):
    """
    Transform the query to produce a better question.
    """
    log.info("*****Start transform the query to produce a better question*****")
    question = state["question"]
    documents = state["documents"]

    better_question = question_rewriter.invoke({"question": question})
    return {"question": better_question, "documents": documents}


if __name__ == "__main__":
    res = generate({"question": "半导体优势是什么", "documents": [Document(page_content="半导体优势是能够提高生产效率和产品质量，降低成本，提高竞争力。")]})
    print(res)