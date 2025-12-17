"""Service for ingesting documents into the knowledge graph."""

from typing import Any

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from scouter.db import get_neo4j_driver, get_neo4j_embedder, get_neo4j_llm


class IngestionService:
    """Service for ingesting documents into Neo4j knowledge graph."""

    def __init__(self) -> None:
        """Initialize the ingestion service with Neo4j connection."""
        self.driver = get_neo4j_driver()
        self.llm = get_neo4j_llm()
        self.embedder = get_neo4j_embedder()

    async def process_document(
        self,
        file_path: str | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a PDF file or text into the knowledge graph using SimpleKGPipeline.

        Args:
            file_path: Path to PDF file to process.
            text: Text content to process.
            metadata: Additional metadata for the document.

        Returns:
            Dictionary containing processing status and type.

        Raises:
            ValueError: If neither file_path nor text is provided.
        """
        if metadata is None:
            metadata = {}

        if file_path is None and text is None:
            msg = "Either file_path or text must be provided"
            raise ValueError(msg)

        try:
            from_pdf = file_path is not None
            kg_builder = SimpleKGPipeline(
                llm=self.llm,
                driver=self.driver,
                embedder=self.embedder,
                from_pdf=from_pdf,
            )
            if from_pdf:
                await kg_builder.run_async(
                    file_path=file_path,
                    document_metadata=metadata,
                )
            else:
                await kg_builder.run_async(text=text, document_metadata=metadata)
        except OSError as e:
            return {"status": "failed", "error": str(e)}
        else:
            return {"status": "processed", "type": "pdf" if from_pdf else "text"}

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()
