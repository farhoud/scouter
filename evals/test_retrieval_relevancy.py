from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from scouter_app.agent.service import SearchService

THRESHOLD = 0.5


def test_retrieval_relevancy(dataset):
    service = SearchService()
    _a, q, _r = dataset

    try:
        results = service.search(q[0])
        context = "\n".join([r.content for r in results])

        metric = ContextualRelevancyMetric()
        test_case = LLMTestCase(
            input=q[0],
            actual_output="",
            retrieval_context=[context],
        )

        metric.measure(test_case)

        assert metric.score is not None
        assert metric.score > THRESHOLD
    finally:
        service.close()
