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

## Ingestion

Ingest documents via the API:
- **PDF**: `POST /v1/ingest` with `file` (multipart/form-data) and optional `metadata` (JSON string).
- **Text**: `POST /v1/ingest` with `text` and optional `metadata`.

Returns a task ID for async processing using Neo4j GraphRAG's SimpleKGPipeline.

## Development Workflow

- **Evals**: Run `uv run eval-watch` for auto-evals on code changes in the `evals/` folder. Use `uv run eval-dev` for dev-specific evals with environment set to development. Evals use a fixture that seeds the DB with PDFs via the ingestion service.
- **Data Prep**: Run `scripts/create_light_subset.py` to generate light datasets for testing.
- **Linting**: `uv run ruff check` and fix issues.

## Environment

Set `SCOUTER_ENV=production` for production deployments. Defaults to 'development'. Affects eval dataset size and logging.

## Examples

- **Chatbot**: See `examples/chatbot/` for a RAG chatbot using OpenRouter.

## Testing

- Run `pytest` for unit tests.
- Use `evals/test_retrieval_relevancy.py` for full retrieval evals (uses service directly).
- `tests/test_mini_eval.py` for fast mini-tests.