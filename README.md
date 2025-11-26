# Project Scouter

Rapid assessment and retrieval from knowledge graph.

## Setup

1. Create virtual environment: `uv venv`
2. Activate: `source .venv/bin/activate`
3. Install: `uv pip install -e .`

## Running

- Terminal 1: `redis-server`
- Terminal 2: `uvicorn app_main:app --reload`
- Terminal 3: `celery -A src.scouter_app.ingestion.tasks worker --loglevel=info`

Or use Docker: `docker-compose up`

API docs at http://localhost:8000/docs