"""Tests for IngestionService."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import scouter_app.ingestion.service as svc
from scouter_app.ingestion.service import IngestionService


@pytest.mark.asyncio
async def test_process_document_text() -> None:
    """Test processing text documents."""
    # Mock the driver and pipeline
    mock_driver = MagicMock()
    mock_llm = MagicMock()
    mock_embedder = MagicMock()
    mock_pipeline = AsyncMock()

    service = IngestionService()
    service.driver = mock_driver
    service.llm = mock_llm
    service.embedder = mock_embedder

    # Mock SimpleKGPipeline
    original_pipeline = svc.SimpleKGPipeline
    svc.SimpleKGPipeline = MagicMock(return_value=mock_pipeline)

    try:
        result = await service.process_document(
            text="sample text",
            metadata={"key": "value"},
        )
        assert result["status"] == "processed"
        assert result["type"] == "text"
        mock_pipeline.run_async.assert_called_once_with(
            text="sample text",
            document_metadata={"key": "value"},
        )
    finally:
        svc.SimpleKGPipeline = original_pipeline


@pytest.mark.asyncio
async def test_process_document_pdf() -> None:
    """Test processing PDF documents."""
    mock_driver = MagicMock()
    mock_llm = MagicMock()
    mock_embedder = MagicMock()
    mock_pipeline = AsyncMock()

    service = IngestionService()
    service.driver = mock_driver
    service.llm = mock_llm
    service.embedder = mock_embedder

    original_pipeline = svc.SimpleKGPipeline
    svc.SimpleKGPipeline = MagicMock(return_value=mock_pipeline)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = temp_file.name

        result = await service.process_document(file_path=temp_path, metadata={})
        assert result["status"] == "processed"
        assert result["type"] == "pdf"
        mock_pipeline.run_async.assert_called_once_with(
            file_path=temp_path,
            document_metadata={},
        )
    finally:
        svc.SimpleKGPipeline = original_pipeline
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
