import argparse
import json
import time
from pathlib import Path

import requests
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from tests.seed_db import seed_db

API_URL = "http://localhost:8000/v1/search"
HTTP_OK = 200


def run_eval(num_docs=5, num_queries=3):
    """
    Run evaluation with configurable parameters.
    """

    # Seed DB with num_docs
    seed_db(num_docs)

    # Wait for indexing
    time.sleep(5)

    # Sample queries (hardcoded for simplicity; in full mode, load from dataset)
    queries = [
        "What is the main contribution of this paper?",
        "Explain the methodology used in the study.",
        "What are the key findings?",
    ][:num_queries]

    # Run queries
    results = []
    for query in queries:
        response = requests.get(API_URL, params={"query": query}, timeout=30)
        if response.status_code == HTTP_OK:
            res_data = response.json()
            context = "\n".join([r["content"] for r in res_data])

            metric = ContextualRelevancyMetric()
            test_case = LLMTestCase(
                input=query,
                actual_output="",
                retrieval_context=[context],
            )
            metric.measure(test_case)
            results.append(
                {
                    "query": query,
                    "score": metric.score,
                    "reason": metric.reason,
                },
            )

    # Save results
    Path("eval_results.json").write_text(json.dumps(results, indent=2))

    return sum(r["score"] for r in results) / len(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Scouter evaluation")
    parser.add_argument("--mode", choices=["mini", "full"], default="mini")
    parser.add_argument("--num-docs", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=3)
    args = parser.parse_args()
    run_eval(args.num_docs, args.num_queries)
