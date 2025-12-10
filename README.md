# Project Scouter

Rapid assessment and retrieval from knowledge graph using Neo4j GraphRAG.

## Overview

Scouter is a knowledge graph-based document retrieval system focused on MCP (Model Context Protocol) for agentic search:

- Ingests PDFs and text documents using Neo4j GraphRAG's SimpleKGPipeline
- Provides agentic semantic search via MCP for LLM integration
- Includes REST API for document ingestion
- Includes evaluation framework for retrieval quality assessment

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)

### Option 1: Docker (Recommended)

```bash
# Start Neo4j with APOC plugin
make neo4j-up

# Start all services (Neo4j, Redis, API, Celery worker)
docker-compose up

# Or run in background
docker-compose up -d
```

### Option 2: Local Development

```bash
# Setup environment
uv venv
source .venv/bin/activate
uv pip install -e .

# Start services (4 terminals)
redis-server
uvicorn app_main:app --reload
celery -A src.scouter_app.ingestion.tasks worker --loglevel=info
make neo4j-up  # Neo4j with APOC
```

## API Usage

### Document Ingestion

```bash
# Ingest PDF
curl -X POST "http://localhost:8000/v1/ingest" \
  -F "file=@document.pdf" \
  -F 'metadata={"source": "user-upload", "doc_id": "doc1"}'

# Ingest text
curl -X POST "http://localhost:8000/v1/ingest" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content", "metadata": {"source": "api"}}'
```

### Interactive API

Visit <http://localhost:8000/docs> for interactive API documentation.

**Note:** Search functionality is provided via MCP (Model Context Protocol) for agentic retrieval. Direct REST search API is not available.

## Architecture

### Components

- **Ingestion Service**: Processes PDFs/text into knowledge graph using SimpleKGPipeline
- **MCP Server**: Core component providing agentic search via Model Context Protocol for LLM integration
- **Celery Workers**: Handle async document processing
- **Redis**: Task queue and caching

### Data Flow

1. Documents → Ingestion API → Celery Queue → Neo4j GraphRAG
2. Search Query → MCP Server → Agentic Search → Neo4j → Ranked Results

## Development

### Code Quality

```bash
# Linting and formatting
uv run ruff check .
uv run ruff format .

# Pre-commit hooks (auto-installed)
git commit  # Hooks run automatically

# Development workflow - watch for changes in src and evals
make eval-watch

# Development workflow with verbose logs
LOGS=1 make eval-watch
```

### Testing

```bash
# Run evaluation tests (with data caching)
make evals

# Force re-ingestion of test data
SCOUTER_FORCE_INGEST=1 make evals

# Run unit tests
pytest
```

### Data Caching

Test data is cached in `.eval_cache/light_subset/` to avoid re-downloading and re-ingesting PDFs across sessions. The fixture:

1. Checks if cached subset matches expected document count
2. Verifies Neo4j contains the ingested documents
3. Skips ingestion if both conditions are met

## Configuration

### Environment Variables

- `SCOUTER_ENV=production` - Production mode (affects eval dataset size and logging)
- `SCOUTER_FORCE_INGEST=1` - Force re-ingestion of test data during evals
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` - Neo4j connection settings
- `REDIS_URL` - Redis connection URL

### Neo4j with APOC

The project uses Neo4j with APOC plugin for enhanced graph procedures. Docker setup automatically installs and configures APOC. For local development, ensure Neo4j has APOC enabled.

## Examples

### MCP Integration (Primary Use Case)

```bash
# Start MCP server
python -m scouter_app.agent.mcp

# Use with Claude Desktop or other MCP-compatible tools
```

Scouter's MCP server enables agentic search for LLMs, providing semantic retrieval from the knowledge graph.

### RAG Chatbot

```bash
cd examples/chatbot
python chatbot.py
```

Interactive chatbot that uses Scouter for retrieval and OpenRouter for generation.

## Project Structure

```
src/scouter_app/
├── agent/          # Search API and MCP server
├── config/         # LLM and client configuration
├── ingestion/      # Document processing and Celery tasks
└── shared/         # Domain models and utilities

evals/              # Evaluation framework and tests
examples/           # Usage examples
scripts/            # Utility scripts
tests/              # Unit tests
```

## Deployment

### Docker Production

```bash
# Build and deploy
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose up -d --scale celery_worker=3
```

### Monitoring

- API health: `GET /health`
- Celery monitoring: Add Flower to docker-compose.yml
- Neo4j Browser: <http://localhost:7474>

## Contributing

1. Fork and create feature branch
2. Make changes with pre-commit hooks ensuring code quality
3. Add tests for new functionality
4. Run `make evals` to verify no regressions
5. Submit pull request

## License

[Add your license here]

## RUN manually

```
CREATE VECTOR INDEX chunkEmbedding IF NOT EXISTS
FOR (m:Chunk)
ON m.embedding
OPTIONS { indexConfig: {
 `vector.dimensions`: 1024, // Qwen/Qwen3-Embedding-0.6B dims
 `vector.similarity_function`: 'cosine'
}}
```

