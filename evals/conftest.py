"""Test configuration and fixtures for evaluation dataset setup."""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from scouter.ingestion.service import IngestionService

from .utils import create_light_subset

logger = logging.getLogger(__name__)


def _check_existing_data(service: IngestionService, total_docs: int) -> bool:
    """Check if data is already ingested in the database.

    Args:
        service: Ingestion service instance.
        total_docs: Expected number of documents.

    Returns:
        True if data is already ingested, False otherwise.
    """
    try:
        with service.driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE n.source = 'light-subset' RETURN count(n) as count",
            )
            record = result.single()
            ingested_count = record["count"] if record else 0
    except OSError as e:
        logger.warning(
            "Failed to check ingested data: %s, proceeding with ingestion",
            e,
        )
        return False
    else:
        return ingested_count >= total_docs


def _load_cached_qa_data(
    cache_dir: Path,
    total_docs: int,
) -> tuple[Any, Any, Any] | None:
    """Load QA data from cache if available.

    Args:
        cache_dir: Cache directory path.
        total_docs: Expected number of documents.

    Returns:
        Tuple of (answers, queries, references) if cache exists and is valid, None otherwise.
    """
    info_file = cache_dir / "info.txt"
    if not info_file.exists():
        return None

    with info_file.open() as f:
        info = f.read()
    if f"Total documents : {total_docs}" not in info:
        return None

    qa_dir = cache_dir / "qa"
    q = json.loads((qa_dir / "queries.json").read_text())
    a = json.loads((qa_dir / "answers.json").read_text())
    r = json.loads((qa_dir / "qrels.json").read_text())
    return a, q, r


def _get_or_create_subset(
    cache_dir: Path,
    total_docs: int,
) -> tuple[Any, Any, Any, Path]:
    """Get existing subset or create new one.

    Args:
        cache_dir: Cache directory path.
        total_docs: Number of documents to create.

    Returns:
        Tuple of (queries, answers, references, subset_dir).
    """
    info_file = cache_dir / "info.txt"

    if info_file.exists():
        with info_file.open() as f:
            info = f.read()
        if f"Total documents : {total_docs}" in info:
            # Reuse cached subset
            qa_dir = cache_dir / "qa"
            q = json.loads((qa_dir / "queries.json").read_text())
            a = json.loads((qa_dir / "answers.json").read_text())
            r = json.loads((qa_dir / "qrels.json").read_text())
            subset_dir = cache_dir
            logger.info("Reusing cached light subset")
            return q, a, r, subset_dir
        # Recreate if count mismatch, clear cache first
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Create new subset
    q, a, r = create_light_subset(
        total_docs=total_docs,
        out_folder=str(cache_dir),
        download_pdf=True,
    )
    subset_dir = cache_dir
    logger.info(
        "Created light subset with %s questions, %s answers, %s references",
        len(q),
        len(a),
        len(r),
    )
    return q, a, r, subset_dir


async def _ingest_documents(service: IngestionService, subset_dir: Path) -> None:
    """Ingest all PDF documents from the subset directory.

    Args:
        service: Ingestion service instance.
        subset_dir: Directory containing PDF files.
    """
    pdf_dir = subset_dir / "pdfs"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info("Starting ingestion of %s PDFs", len(pdf_files))

    tasks = []
    for i, pdf_path in enumerate(pdf_files, 1):
        doc_id = pdf_path.stem
        logger.info("Ingesting document %s/%s: %s", i, len(pdf_files), doc_id)
        tasks.append(
            service.process_document(
                file_path=str(pdf_path),
                metadata={"source": "light-subset", "doc_id": doc_id},
            ),
        )

    # Wait for all ingestions to complete
    await asyncio.gather(*tasks)
    logger.info("Ingestion completed")


@pytest.fixture(scope="session", autouse=True)
async def dataset() -> tuple[Any, Any, Any]:
    """Fixture to seed the database with a light subset of documents using the ingestion service.

    Downloads PDFs, ingests them, and waits for completion.
    Re-seeds if ingestion package has changed.

    Returns:
        Tuple containing answers, queries, and references data.
    """
    logger.info("Starting dataset fixture: creating light subset")

    cache_dir = Path(".eval_cache") / "light_subset"
    total_docs = 5
    force_ingest = os.getenv("SCOUTER_FORCE_INGEST") == "1"

    service = IngestionService()

    # Check if data already ingested
    if not force_ingest and _check_existing_data(service, total_docs):
        logger.info("Data already ingested, reusing")
        cached_data = _load_cached_qa_data(cache_dir, total_docs)
        if cached_data:
            service.close()
            return cached_data
        logger.warning("Data ingested but cache missing, proceeding to create cache")

    # Get or create subset
    q, a, r, subset_dir = _get_or_create_subset(cache_dir, total_docs)

    # Ingest documents
    await _ingest_documents(service, subset_dir)

    service.close()
    return a, q, r
