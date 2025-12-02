"""Main FastAPI application for Project Scouter."""

import logging

from fastapi import FastAPI

from src.scouter_app.agent.api import router as agent_router
from src.scouter_app.agent.mcp import app as mcp_app
from src.scouter_app.config.llm import get_client_config
from src.scouter_app.ingestion.api import router as ingestion_router

logger = logging.getLogger(__name__)

config = get_client_config()
logger.info("Starting Scouter in %s environment", config.env)

app: FastAPI = FastAPI(
    title="Project Scouter",
    description="Rapid assessment and retrieval from knowledge graph",
)

# Include REST API routers
app.include_router(ingestion_router)
app.include_router(agent_router)

# Mount FastMCP for tool access
app.mount("/mcp", mcp_app)  # type: ignore[arg-type]
