from src.agent.graph import graph
from utils.log_utils import log
import uuid
from utils.print_utils import _print_event

# draw_graph(graph, 'graph_rag1-2.png')
config = {
    "configurable": {
        # 检查点由session_id访问
        "thread_id": str(uuid.uuid4()),
    }
}

_printed = set()  # set集合，避免重复打印

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
                _print_event(event, _printed)