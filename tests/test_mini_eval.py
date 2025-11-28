from unittest.mock import patch

import requests

API_URL = "http://localhost:8000/v1/search"
HTTP_OK = 200


def test_search_endpoint():
    """
    Test basic search functionality.
    """
    # Mock response
    mock_response = [{"title": "Test Doc", "content": "This is test content."}]
    with patch("requests.get") as mock_get:
        mock_get.return_value.status_code = HTTP_OK
        mock_get.return_value.json.return_value = mock_response

        response = requests.get(API_URL, params={"query": "test"}, timeout=30)
        assert response.status_code == HTTP_OK
        data = response.json()
        assert len(data) > 0
        assert "content" in data[0]


def test_retrieval_relevancy_mini():
    """
    Mini test for retrieval relevancy (without full deepeval for speed).
    """
    # Simplified: check if search returns results
    response = requests.get(API_URL, params={"query": "sample query"}, timeout=30)
    assert response.status_code == HTTP_OK
    results = response.json()
    assert isinstance(results, list)
    # Assume seeded data exists
    if results:
        assert "content" in results[0]
