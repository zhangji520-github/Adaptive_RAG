import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agent.graph import graph

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_agent_simple_passthrough() -> None:
    # 使用符合GraphState要求的输入格式
    inputs = {
        "question": "What is the capital of France?",
        "generation": "",
        "documents": []
    }
    res = await graph.ainvoke(inputs)
    assert res is not None
    # 验证返回结果包含必要的字段
    assert "question" in res
    assert "generation" in res
