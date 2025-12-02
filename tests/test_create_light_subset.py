"""Tests for create_light_subset script."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from scripts.create_light_subset import create_light_subset


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for testing."""
    with patch("scripts.create_light_subset.requests.get") as mock_get:
        # Mock pdf_urls.json
        mock_pdf_urls = Mock()
        mock_pdf_urls.json.return_value = [
            {"paper_id": "paper1", "url": "http://example.com/paper1.pdf"},
            {"paper_id": "paper2", "url": "http://example.com/paper2.pdf"},
            {"paper_id": "paper3", "url": "http://example.com/paper3.pdf"},
            {"paper_id": "paper4", "url": "http://example.com/paper4.pdf"},
        ]
        mock_qrels = Mock()
        mock_qrels.json.return_value = {
            "q1": [{"doc_id": "paper1"}],
            "q2": [{"doc_id": "paper1"}],
        }
        mock_queries = Mock()
        mock_queries.json.return_value = {
            "q1": "What is it?",
            "q2": "How does it work?",
        }
        mock_answers = Mock()
        mock_answers.json.return_value = {"q1": "answer1", "q2": "answer2"}
        mock_get.side_effect = [
            mock_pdf_urls,  # pdf_urls.json
            mock_qrels,  # qrels.json
            mock_queries,  # queries.json
            mock_answers,  # answers.json
            Mock(content=b"corpus1"),  # corpus/paper1.json
            Mock(content=b"corpus2"),  # corpus/paper2.json
            Mock(content=b"corpus3"),  # corpus/paper3.json
        ]
        yield mock_get


def test_create_light_subset(tmp_path: Path, mock_requests_get: Mock) -> None:
    """Test create_light_subset function with mocked requests."""
    expected_docs = 3
    out_folder = str(tmp_path / "light_subset")
    create_light_subset(
        total_docs=expected_docs,
        out_folder=out_folder,
        download_pdf=False,
    )

    # Check directories created
    assert (tmp_path / "light_subset" / "pdfs").exists()
    assert (tmp_path / "light_subset" / "corpus").exists()
    assert (tmp_path / "light_subset" / "qa").exists()

    # Check corpus files
    corpus_dir = tmp_path / "light_subset" / "corpus"
    corpus_files = list(corpus_dir.glob("*.json"))
    assert len(corpus_files) == expected_docs
    assert any("paper1.json" in str(f) for f in corpus_files)
    with (corpus_dir / "paper1.json").open("rb") as f:
        assert f.read() == b"corpus1"

    # Check qa files
    qa_dir = tmp_path / "light_subset" / "qa"
    queries = json.loads((qa_dir / "queries.json").read_text())
    answers = json.loads((qa_dir / "answers.json").read_text())
    qrels = json.loads((qa_dir / "qrels.json").read_text())

    assert queries == {"q1": "What is it?", "q2": "How does it work?"}
    assert answers == {"q1": "answer1", "q2": "answer2"}
    assert qrels == {"q1": [{"doc_id": "paper1"}], "q2": [{"doc_id": "paper1"}]}

    # Check info.txt
    info = (tmp_path / "light_subset" / "info.txt").read_text()
    assert "Light Open-RAG-Bench subset" in info
    assert f"Total documents : {expected_docs}" in info

    # Check that the correct URLs were called
    base_url = "https://huggingface.co/datasets/vectara/open_ragbench/raw/main/official"
    expected_urls = [
        f"{base_url}/pdf/arxiv/pdf_urls.json",
        f"{base_url}/qa/arxiv/qrels.json",
        f"{base_url}/qa/arxiv/queries.json",
        f"{base_url}/qa/arxiv/answers.json",
        f"{base_url}/pdf/arxiv/corpus/paper1.json",
        f"{base_url}/pdf/arxiv/corpus/paper2.json",
        f"{base_url}/pdf/arxiv/corpus/paper3.json",
    ]
    called_urls = [call[0][0] for call in mock_requests_get.call_args_list]
    assert called_urls == expected_urls
