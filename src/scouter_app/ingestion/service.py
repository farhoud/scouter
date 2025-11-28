import os

from neo4j import GraphDatabase


class IngestionService:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def ingest_document(self, title: str, content: str, metadata: dict):
        """
        Ingest a document into the knowledge graph.
        """
        with self.driver.session() as session:
            session.run(
                "CREATE (d:Document {title: $title, content: $content, metadata: $metadata})",
                title=title,
                content=content,
                metadata=metadata,
            )
        return {"status": "ingested", "title": title}

    def close(self):
        self.driver.close()
