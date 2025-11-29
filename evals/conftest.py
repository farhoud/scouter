import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to sys.path to import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.create_light_subset import create_light_subset
from scouter_app.ingestion.service import IngestionService

INGESTION_DIR = Path("src/scouter_app/ingestion")
CACHE_FILE = Path(".eval_cache")


@pytest.fixture(scope="session", autouse=True)
async def dataset():
    """
    Fixture to seed the database with a light subset of documents using the ingestion service.
    Downloads PDFs, ingests them, and waits for completion.
    Re-seeds if ingestion package has changed.
    """

    # Create a temporary directory for the subset
    with tempfile.TemporaryDirectory() as temp_dir:
        subset_dir = Path(temp_dir) / "light_subset"
        subset_dir.mkdir()

        # Create light subset with 5 docs
        q, a, r = create_light_subset(
            total_docs=5, out_folder=str(subset_dir), download_pdf=True
        )

        # Ingest each PDF
        pdf_dir = subset_dir / "pdfs"
        service = IngestionService()
        tasks = []
        for pdf_path in pdf_dir.glob("*.pdf"):
            doc_id = pdf_path.stem
            tasks.append(
                service.process_document(
                    file_path=str(pdf_path),
                    metadata={"source": "light-subset", "doc_id": doc_id},
                )
            )

        # Wait for all ingestions to complete
        await asyncio.gather(*tasks)
        service.close()

        # Update cache
        # Yield to allow tests to run
        return a, q, r

        # Cleanup if needed (but temp dir will be cleaned up)
