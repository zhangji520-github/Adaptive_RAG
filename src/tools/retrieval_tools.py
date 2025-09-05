from env_utils import MILVUS_URI, COLLECTION_NAME
from langchain_milvus import Milvus, BM25BuiltInFunction
from llm_utils import openai_embedding
from utils.log_utils import log


def create_vectorstore():
    """创建向量存储，确保在正确的上下文中初始化"""
    def _create_milvus_instance():
        """内部函数：创建 Milvus 实例"""
        return Milvus(
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
    
    try:
        vectorstore = _create_milvus_instance()
        log.info("建立连接数据库成功")
        return vectorstore
    except Exception as e:
        log.warning(f"初始化向量存储时遇到警告: {e}")
        # 在某些异步环境中，可能需要重试连接
        try:
            vectorstore = _create_milvus_instance()
            log.info("重试后成功建立连接数据库")
            return vectorstore
        except Exception as retry_error:
            log.error(f"重试后仍然失败: {retry_error}")
            raise retry_error

# 创建向量存储实例
vectorstore = create_vectorstore()

# 创建一个更宽松的检索器配置，避免过滤条件过于严格导致空结果
def create_retriever():
    """创建检索器，使用更宽松的配置确保有结果返回"""
    try:
        # 首先尝试使用过滤条件
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "ranker_type": "rrf",          # 使用支持的ranker类型 "rrf" 
                "ranker_params": {"k":60},     # rrf算法的参数k值
                "expr": 'category == "TitleWithContent" && category_depth > 1'
            }
        )
        log.info("使用过滤条件创建检索器成功")
        return retriever
    except Exception as e:
        log.warning(f"使用过滤条件创建检索器失败，使用备用配置: {e}")
        # 如果过滤条件有问题，使用更简单的配置
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "ranker_type": "rrf",
                "ranker_params": {"k":60}
                # 移除过滤条件
            }
        )
        log.info("使用备用配置创建检索器成功")
        return retriever

retriever = create_retriever()

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    res = retrieve({"question": "半导体优势是什么"})
    print(res)