import subprocess
import sys
import time

import pytest
import requests
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

# Our API endpoint
API_URL = "http://localhost:8000/v1/search"
INGEST_API = "http://localhost:8000/v1/ingest"
HTTP_OK = 200
SCORE_THRESHOLD = 0.5


@pytest.fixture(scope="session", autouse=True)
def seed_database():
    # Start services if not running (assume docker-compose or manual)
    # For simplicity, assume running

    # Run seed_db.py
    subprocess.run([sys.executable, "tests/seed_db.py"], check=True)
    # Wait for ingestion to complete
    time.sleep(10)  # Adjust as needed


@pytest.mark.parametrize(
    "query",
    [
        "What is the main contribution of this paper?",
        "Explain the methodology used in the study.",
        "What are the key findings?",
        # Add more queries from dataset
    ],
)
def test_retrieval_relevancy(query):
    metric = ContextualRelevancyMetric()

    response = requests.get(API_URL, params={"query": query}, timeout=30)
    assert response.status_code == HTTP_OK
    results = response.json()
    context = "\n".join([r["content"] for r in results])

    test_case = LLMTestCase(
        input=query,
        actual_output="",  # Not used
        retrieval_context=[context],
    )

    metric.measure(test_case)

    assert metric.score > SCORE_THRESHOLD  # Example threshold
