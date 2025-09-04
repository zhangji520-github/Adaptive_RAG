from env_utils import MILVUS_URI, COLLECTION_NAME
from langchain_milvus import Milvus, BM25BuiltInFunction
from llm_utils import openai_embedding
from utils.log_utils import log
from langchain.tools.retriever import create_retriever_tool

# 使用langchain_milvus 建立连接数据库
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
log.info("建立连接数据库成功")

# 方法2：如果需要混合检索并添加过滤条件，可以这样设置：
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "ranker_type": "rrf",          # 使用支持的ranker类型 "rrf" 
        "ranker_params": {"k":60},     # rrf算法的参数k值
        "expr": 'category == "TitleWithContent" && category_depth > 1'
    }
)

retrieval_tool = create_retriever_tool(
    retriever,
    name="retrieval_tool",
    description="用于检索并返回有关‘半导体和芯片’的信息，内容包含：半导体和芯片的封装测试等"
)

if __name__ == "__main__":
    res = retrieval_tool.invoke("半导体和芯片的优势")
    print(res)