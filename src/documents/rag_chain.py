import sys
import os

from sympy import vector
# 添加上级目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from llm_utils import qwen_embeddings, openai_embedding, llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.prompts import PromptTemplate
from markdown_parser import MarkdownParser
from langchain_milvus import Milvus, BM25BuiltInFunction
from env_utils import MILVUS_URI, COLLECTION_NAME
# Define the prompt template for generating AI responses
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and you provide answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""

# Create a PromptTemplate instance with the defined template and input variables
prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

class RagChain:
    """自定义ragchain"""
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def run_chain(self, retrieval, question):
        # 首先获取检索结果
        retrieved_docs = retrieval.invoke(question)
        
        # 打印检索结果
        print("🔍 检索结果:")
        print("=" * 40)
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"文档片段 {i}:")
            print(f"内容: {doc.page_content}")
            print("-" * 30)
        
        # 构建RAG链
        rag_chain = (
            {"context": retrieval | RagChain.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # 生成答案
        print("\n🤖 生成答案:")
        # res = rag_chain.invoke(question)
        # print(res)
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)

def create_milvus_connection():
    """连接到已经创建好的 Milvus 向量数据库"""
    vectorstore = Milvus(
        embedding_function=openai_embedding,
        collection_name=COLLECTION_NAME,
        connection_args={
            "uri": MILVUS_URI,
        },
        # 自动将文本字段（TEXT_FIELD）通过 内置的 BM25 函数 转换为 稀疏向量（SPARSE_FLOAT_VECTOR）
        builtin_function=BM25BuiltInFunction( 
            input_field_names="text",      # 输入：原始文本字段 	VARCHAR
            output_field_names="sparse",   # 输出：稀疏向量字段，对应 vector_field[0] SPARSE_FLOAT_VECTOR
        ), 
        vector_field=["sparse", "dense"],  # 指定要存储的向量字段 dense 用于 OpenAI 嵌入， sparse 用于 BM25 函数。
        consistency_level="Strong",          # 一致性级别
        drop_old=False,                       # 是否删除已有的 collection
    )
    return vectorstore


if __name__ == "__main__":
    # # 1. 加载文档数据
    # file_path = r"E:\Workspace\ai\RAG\datas\md\tech_report_z7tx05vt.md"
    # parser = MarkdownParser()
    # docs = parser.parse_markdown_to_documents(file_path)
    # print(f"加载了 {len(docs)} 个文档块")

    # 2. 创建 Milvus 向量数据库并添加文档 
    vectorstore = create_milvus_connection()

    # 3. 创建 RAG 链并测试
    rag = RagChain()
    
    
    # 方法2：如果需要混合检索并添加过滤条件，可以这样设置：
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
            "ranker_type": "weighted",          # 也可以使用 "rrf" 这样的话params设置 {"k":100}
            "ranker_params": {"weights": [0.3, 0.7]},  # sparse权重0.3，dense权重0.7
            "expr": 'category == "TitleWithContent" && category_depth > 1'
        }
    )
    

    
    # 5. 测试问题
    test_questions = [
        "半导体制造中的虚拟计量技术？",
    ]
    
    print("\n开始测试混合检索:")
    # 移除 output_fields 参数以避免与混合搜索的参数冲突  如果想要更加精细的控制 可以用 客户端milvus自己提供的原生 hybrid_search 方法 自己写 ANNSearchRequest
    results = vectorstore.similarity_search_with_score(
        "干法刻蚀的优势", 
        k=4, 
        ranker_type="weighted", 
        ranker_params={"weights": [0.3, 0.7]},
        expr='category == "TitleWithContent" && category_depth > 1'
    )
    print(f"检索到 {len(results)} 个结果:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"内容: {doc.page_content}")
        print(f"相似度分数: {score}")
        print("-" * 30)

    # print("\n开始测试 RAG 系统:")
    # print("=" * 50)
    
    # for i, question in enumerate(test_questions, 1):
    #     print(f"\n问题 {i}: {question}")
    #     print("-" * 30)
        
    #     try:
    #         rag.run_chain(retriever, question)
    #     except Exception as e:
    #         print(f"错误: {e}")
        
    #     print("-" * 50)
