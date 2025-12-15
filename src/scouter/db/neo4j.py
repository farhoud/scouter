"""Database connections and utilities for Scouter.

This module provides Neo4j driver setup and related database utilities.
"""

import os

from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.llm import OpenAILLM

import neo4j
from neo4j import GraphDatabase


def get_neo4j_driver() -> neo4j.Driver:
    """Get a Neo4j driver instance."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    return GraphDatabase.driver(uri, auth=(user, password))


def get_neo4j_llm() -> OpenAILLM:
    """Get a Neo4j LLM instance configured for OpenAI."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY environment variable is required"
        raise ValueError(msg)
    return OpenAILLM(model_name="gpt-4o-mini", model_params={"api_key": api_key})


def get_neo4j_embedder() -> SentenceTransformerEmbeddings:
    """Get a Neo4j embedder instance."""
    return SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
