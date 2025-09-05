from typing import TypedDict, List


# 这里我们单独用一个question存储问题，免得还要额外写函数从Message里面的content拿到question
class GraphState(TypedDict):
    """
    Represents the state of the graph

    Attributes:
        question: The question to be answered
        generation: Last generation of the question
        documents: Documents retrieved from the vector database
    """
    question: str
    generation: str
    documents: List[str]