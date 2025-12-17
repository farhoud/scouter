"""Database connections and utilities for Scouter.

This module provides Neo4j driver setup and related database utilities.
"""

from functools import lru_cache

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.llm import OpenAILLM

from neo4j import GraphDatabase
from scouter.config import config


@lru_cache(maxsize=1)
def get_neo4j_driver():
    """Get a singleton Neo4j driver instance."""
    return GraphDatabase.driver(
        config.db.uri, auth=(config.db.user, config.db.password)
    )


@lru_cache(maxsize=1)
def get_neo4j_llm():
    """Get a singleton Neo4j LLM instance."""
    return OpenAILLM(
        config.db.llm_model, api_key=config.llm.api_key, base_url=config.llm.base_url
    )


@lru_cache(maxsize=1)
def get_neo4j_embedder():
    """Get a singleton Neo4j embedder instance."""
    return SentenceTransformerEmbeddings(config.db.embedder_model)
