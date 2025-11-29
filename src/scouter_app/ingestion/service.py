import os
from typing import Optional

from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from scouter_app.config.llm import get_neo4j_embedder, get_neo4j_llm


class IngestionService:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        # Singleton instances
        self.llm = get_neo4j_llm()
        self.embedder = get_neo4j_embedder()

    async def process_document(
        self,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        metadata: dict = {},
    ):
        """
        Process a PDF file or text into the knowledge graph using SimpleKGPipeline.
        """
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
                    file_path=file_path, document_metadata=metadata
                )
            else:
                await kg_builder.run_async(text=text, document_metadata=metadata)
            return {"status": "processed", "type": "pdf" if from_pdf else "text"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def close(self):
        self.driver.close()
