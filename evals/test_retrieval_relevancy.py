import pytest
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from examples.chatbot.chatbot import chat_with_rag

THRESHOLD = 0.5


@pytest.mark.parametrize(
    "query_index", range(5)
)  # Assuming 5 queries based on total_docs
def test_chatbot_answer_relevancy(dataset, query_index):
    _a, q, _r = dataset

    queries = list(q.values())
    query_data = queries[query_index]
    query = query_data["query"]
    response = chat_with_rag(query)

    metric = AnswerRelevancyMetric()
    test_case = LLMTestCase(
        input=query,
        actual_output=response,
    )

    metric.measure(test_case)

    assert metric.score is not None
    assert metric.score > THRESHOLD
