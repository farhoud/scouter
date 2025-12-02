from unittest.mock import AsyncMock, MagicMock

import pytest

from scouter_app.ingestion.service import IngestionService


@pytest.mark.asyncio
async def test_process_document_text():
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
    import scouter_app.ingestion.service as svc

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
async def test_process_document_pdf():
    mock_driver = MagicMock()
    mock_llm = MagicMock()
    mock_embedder = MagicMock()
    mock_pipeline = AsyncMock()

    service = IngestionService()
    service.driver = mock_driver
    service.llm = mock_llm
    service.embedder = mock_embedder

    import scouter_app.ingestion.service as svc

    original_pipeline = svc.SimpleKGPipeline
    svc.SimpleKGPipeline = MagicMock(return_value=mock_pipeline)

    try:
        result = await service.process_document(file_path="/tmp/test.pdf", metadata={})
        assert result["status"] == "processed"
        assert result["type"] == "pdf"
        mock_pipeline.run_async.assert_called_once_with(
            file_path="/tmp/test.pdf",
            document_metadata={},
        )
    finally:
        svc.SimpleKGPipeline = original_pipeline
