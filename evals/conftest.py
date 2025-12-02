import asyncio
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pytest

# Add parent directory to sys.path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from scouter_app.ingestion.service import IngestionService
from scripts.create_light_subset import create_light_subset

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
async def dataset():
    """
    Fixture to seed the database with a light subset of documents using the ingestion service.
    Downloads PDFs, ingests them, and waits for completion.
    Re-seeds if ingestion package has changed.
    """

    logger.info("Starting dataset fixture: creating light subset")

    cache_dir = Path(".eval_cache") / "light_subset"
    info_file = cache_dir / "info.txt"
    total_docs = 5

    # Check if force re-ingest
    force_ingest = os.getenv("SCOUTER_FORCE_INGEST") == "1"

    service = IngestionService()

    # Check if data already ingested
    if not force_ingest:
        try:
            with service.driver.session() as session:
                result = session.run(
                    "MATCH (n) WHERE n.source = 'light-subset' RETURN count(n) as count",
                )
                record = result.single()
                ingested_count = record["count"] if record else 0
            if ingested_count >= total_docs:
                logger.info(
                    f"Data already ingested ({ingested_count} documents), reusing",
                )
                # Load QA data from cache if available
                if info_file.exists():
                    with info_file.open() as f:
                        info = f.read()
                    if f"Total documents : {total_docs}" in info:
                        qa_dir = cache_dir / "qa"
                        q = json.loads((qa_dir / "queries.json").read_text())
                        a = json.loads((qa_dir / "answers.json").read_text())
                        r = json.loads((qa_dir / "qrels.json").read_text())
                        service.close()
                        return a, q, r
                # If no cache, still skip ingestion but need to return something? Wait, if ingested, but no cache, perhaps error or create cache.
                # For now, assume if ingested, cache exists.
                logger.warning(
                    "Data ingested but cache missing, proceeding to create cache",
                )
        except Exception as e:
            logger.warning(
                f"Failed to check ingested data: {e}, proceeding with ingestion",
            )

    # Proceed with subset creation/loading
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
        else:
            # Recreate if count mismatch, clear cache first
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            q, a, r = create_light_subset(
                total_docs=total_docs,
                out_folder=str(cache_dir),
                download_pdf=True,
            )
            subset_dir = cache_dir
            logger.info(
                f"Recreated light subset with {len(q)} questions, {len(a)} answers, {len(r)} references",
            )
    else:
        # Create new subset
        q, a, r = create_light_subset(
            total_docs=total_docs,
            out_folder=str(cache_dir),
            download_pdf=True,
        )
        subset_dir = cache_dir
        logger.info(
            f"Created light subset with {len(q)} questions, {len(a)} answers, {len(r)} references",
        )

    # Ingest each PDF
    pdf_dir = subset_dir / "pdfs"
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Starting ingestion of {len(pdf_files)} PDFs")

    tasks = []
    for i, pdf_path in enumerate(pdf_files, 1):
        doc_id = pdf_path.stem
        logger.info(f"Ingesting document {i}/{len(pdf_files)}: {doc_id}")
        tasks.append(
            service.process_document(
                file_path=str(pdf_path),
                metadata={"source": "light-subset", "doc_id": doc_id},
            ),
        )

    # Wait for all ingestions to complete
    await asyncio.gather(*tasks)
    logger.info("Ingestion completed")
    service.close()

    # Yield to allow tests to run
    return a, q, r
