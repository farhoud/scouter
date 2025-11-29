import os

from neo4j import GraphDatabase

from scouter_app.shared.domain_models import SearchResult


class SearchService:
    def __init__(self):
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """
        Search the knowledge graph for relevant content.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (d:Document)
                WHERE d.content CONTAINS $search_query OR d.title CONTAINS $search_query
                RETURN d.title as title, d.content as content, id(d) as node_id
                LIMIT $limit
                """,
                search_query=query,
                limit=limit,
            )
            records = [record for record in result]
            return [
                SearchResult(
                    content=f"{record['title']}: {record['content'][:200]}...",
                    score=1.0,  # Simplified score
                    node_id=str(record["node_id"]),
                )
                for record in records
            ]

    def close(self):
        self.driver.close()
