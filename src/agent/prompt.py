REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "Score the document as 'yes' if it meets ANY of these criteria:\n"
    "1. Contains direct keywords from the question\n"
    "2. Discusses the same topic or domain (e.g., 半导体, 芯片, 封装, 测试等)\n"
    "3. Provides any information that could partially answer the question\n"
    "4. Contains technical concepts related to the question's subject area\n"
    "Be GENEROUS in your evaluation - if there's any reasonable connection, score as 'yes'.\n"
    "Only score 'no' if the content is completely unrelated to the question domain.\n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)