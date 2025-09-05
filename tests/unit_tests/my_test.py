from src.agent.graph import graph
from utils.log_utils import log
import uuid

# draw_graph(graph, 'graph_rag1-2.png')
config = {
    "configurable": {
        # 检查点由session_id访问
        "thread_id": str(uuid.uuid4()),
    }
}

_printed = set()  # set集合，避免重复打印

def print_graph_state_event(event: dict, _printed: set, max_length=1500):
    """
    打印 GraphState 事件信息
    
    参数:
        event (dict): GraphState 事件字典，包含 question, generation, documents
        _printed (set): 已打印内容的集合，用于避免重复打印
        max_length (int): 内容的最大长度，超过此长度将被截断
    """
    if not event:
        return
        
    generation = event.get("generation", "")
    documents = event.get("documents", [])
    
    # 策略：
    # 1. 有generation时：只打印回答（generate节点）
    # 2. 有documents但无generation时：只打印文档（retrieve/web_search节点）
    
    if generation:
        # 生成节点：只显示最终回答
        generation_id = f"generation_{hash(generation)}"
        if generation_id not in _printed:
            print("=" * 50)
            generation_display = generation
            if len(generation_display) > max_length:
                generation_display = generation_display[:max_length] + " ... （已截断）"
            print(f"🤖 回答: {generation_display}")
            print("=" * 50)
            _printed.add(generation_id)
    
    elif documents:
        # 检索/搜索节点：显示检索到的文档
        docs_content = []
        for doc in documents:
            if hasattr(doc, 'page_content'):
                docs_content.append(doc.page_content)
            elif isinstance(doc, str):
                docs_content.append(doc)
            else:
                docs_content.append(str(doc))
        
        docs_hash = hash(tuple(docs_content))
        docs_id = f"documents_{docs_hash}"
        
        if docs_id not in _printed:
            print("=" * 50)
            print(f"📚 相关文档数量: {len(documents)}")
            print("📄 检索到的文档内容:")
            for i, doc in enumerate(documents, 1):
                print(f"\n--- 文档 {i} ---")
                if hasattr(doc, 'page_content'):
                    # 处理 Document 对象（来自向量数据库检索）
                    content = doc.page_content
                    if len(content) > 300:  # 限制单个文档显示长度
                        content = content[:300] + "... (内容已截断)"
                    print(f"内容: {content}")
                    
                    if hasattr(doc, 'metadata') and doc.metadata:
                        print(f"元数据: {doc.metadata}")
                elif isinstance(doc, str):
                    # 处理字符串格式（来自网络搜索）
                    content_display = doc
                    if len(content_display) > 300:
                        content_display = content_display[:300] + "... (内容已截断)"
                    print(f"内容: {content_display}")
                elif isinstance(doc, dict):
                    # 处理字典格式（可能来自API返回）
                    print(f"内容: {str(doc)[:300]}{'... (内容已截断)' if len(str(doc)) > 300 else ''}")
                else:
                    # 处理其他未知格式
                    print(f"内容: {str(doc)[:300]}{'... (内容已截断)' if len(str(doc)) > 300 else ''}")
            print("=" * 50)
            _printed.add(docs_id)

# 测试graph
if __name__ == "__main__":
    # 执行工作流
    while True:
        question = input('用户：')
        if question.lower() in ['q', 'exit', 'quit']:
            log.info('对话结束，拜拜！')
            break
        else:
            inputs = {
                "question": question,
                "generation": "",
                "documents": []
            }
            events = graph.stream(inputs, config=config, stream_mode='values')
            # 打印消息
            for event in events:
                print_graph_state_event(event, _printed)