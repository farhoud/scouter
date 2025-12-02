# Project Scouter

Rapid assessment and retrieval from knowledge graph.

## Setup

1. Create virtual environment: `uv venv`
2. Activate: `source .venv/bin/activate`
3. Install: `uv pip install -e .`

## Running

### Option 1: Docker (Recommended)
```bash
make neo4j-up  # Starts Neo4j with APOC enabled
docker-compose up  # Starts all services
```

### Option 2: Local Development
- Terminal 1: `redis-server`
- Terminal 2: `uvicorn app_main:app --reload`
- Terminal 3: `celery -A src.scouter_app.ingestion.tasks worker --loglevel=info`
- Terminal 4: `make neo4j-up` (for Neo4j with APOC)

API docs at http://localhost:8000/docs

## Ingestion

Ingest documents via the API:
- **PDF**: `POST /v1/ingest` with `file` (multipart/form-data) and optional `metadata` (JSON string).
- **Text**: `POST /v1/ingest` with `text` and optional `metadata`.

Returns a task ID for async processing using Neo4j GraphRAG's SimpleKGPipeline.

### Neo4j with APOC
The project uses Neo4j with APOC plugin enabled for enhanced graph procedures. When using Docker, APOC is automatically installed and configured. For local development, ensure Neo4j has APOC enabled.

## Development Workflow

- **Evals**: 
  - `make evals` - Run evaluation tests (reuses cached data)
  - `SCOUTER_FORCE_INGEST=1 make evals` - Force re-ingestion of test data
  - Evals use a fixture that seeds the DB with PDFs via the ingestion service and caches data in `.eval_cache/` for reuse across sessions.
- **Data Prep**: Run `scripts/create_light_subset.py` to generate light datasets for testing.
- **Linting**: `uv run ruff check` and fix issues.

## Environment

- `SCOUTER_ENV=production` - Production mode (affects eval dataset size and logging)
- `SCOUTER_FORCE_INGEST=1` - Force re-ingestion of test data during evals
- Defaults to 'development'.

## Examples

- **Chatbot**: See `examples/chatbot/` for a RAG chatbot using OpenRouter.

## Testing

- `make evals` - Run evaluation tests with cached data
- `pytest` - Run unit tests
- `evals/test_retrieval_relevancy.py` - Full retrieval evals (uses service directly)
- `tests/test_mini_eval.py` - Fast mini-tests

### Data Caching
Test data is cached in `.eval_cache/light_subset/` to avoid re-downloading and re-ingesting PDFs across test sessions. The fixture checks:
1. If cached subset matches expected document count
2. If Neo4j already contains the ingested documents
3. Skips ingestion if both conditions are met