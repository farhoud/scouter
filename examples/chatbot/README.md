# Scouter Chatbot Example

A simple CLI chatbot that uses Scouter for retrieval-augmented generation (RAG) with OpenRouter as the LLM provider.

## Setup

1. Install dependencies:
   ```
   uv pip install -e .[dev]
   ```

2. Set environment variables:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key.
   - Ensure Scouter is running (e.g., `uvicorn app_main:app --reload`).

3. Run the chatbot:
   ```
   python chatbot.py
   ```

## Usage

- Type questions in the CLI.
- The bot retrieves relevant context from Scouter's knowledge graph.
- Generates responses using OpenRouter's models.

## Integration

- Uses Scouter's `/v1/search` for context retrieval.
- Can be extended for eval (e.g., measure response quality).