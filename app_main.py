"""Main FastAPI application for Project Scouter."""

import logging

from fastapi import FastAPI

from src.scouter.agent.mcp import app as mcp_app
from src.scouter.config.llm import get_client_config
from src.scouter.config.logging import setup_logging
from src.scouter.ingestion.api import router as ingestion_router

# Setup logging
setup_logging()

logger = logging.getLogger(__name__)

config = get_client_config()
logger.info("Starting Scouter in %s environment", config.env)

app: FastAPI = FastAPI(
    title="Project Scouter",
    description="Rapid assessment and retrieval from knowledge graph",
)

# Include REST API routers
app.include_router(ingestion_router)

# Mount FastMCP for tool access
app.mount("/mcp", mcp_app)  # type: ignore[arg-type]
