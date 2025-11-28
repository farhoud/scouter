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

## Development Workflow

- **Eval Scripts**: Use `scripts/run_eval.py` for configurable evals or `scripts/watch_eval.py` for auto-evals on code changes.
- **Data Prep**: Run `scripts/create_light_subset.py` to generate light datasets for testing.
- **Linting**: `uv run ruff check` and fix issues.

## Examples

- **Chatbot**: See `examples/chatbot/` for a RAG chatbot using OpenRouter.

## Testing

- Run `pytest` for unit tests.
- Use `tests/test_eval.py` for full retrieval evals.
- `tests/test_mini_eval.py` for fast mini-tests.