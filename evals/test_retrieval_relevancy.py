import json
from pathlib import Path

from neo4j._work import query
import pytest
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from scouter_app.agent.service import SearchService


def test_retrieval_relevancy(dataset):
    service = SearchService()
    a, q, r = dataset

    try:
        results = service.search(q[0])
        context = "\n".join([r.content for r in results])

        metric = ContextualRelevancyMetric()
        test_case = LLMTestCase(
            input=q[0],
            actual_output="",  # Not used
            retrieval_context=[context],
        )

        metric.measure(test_case)

        assert metric.score is not None and metric.score > 0.5  # Example threshold
    finally:
        service.close()
