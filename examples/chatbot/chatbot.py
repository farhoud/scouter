import os

import openai
import requests

HTTP_OK = 200

# Set OpenRouter API key
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

SCOUTER_SEARCH_URL = "http://localhost:8000/v1/search"


def retrieve_context(query: str) -> str:
    """
    Retrieve context from Scouter.
    """
    response = requests.get(SCOUTER_SEARCH_URL, params={"query": query}, timeout=30)
    if response.status_code == HTTP_OK:
        results = response.json()
        return "\n".join([r["content"] for r in results])
    return "No context retrieved."


def chat_with_rag():
    """
    CLI chatbot with RAG using Scouter + OpenRouter.
    """

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        # Retrieve context
        context = retrieve_context(user_input)

        # Build prompt
        prompt = f"Context: {context}\n\nQuestion: {user_input}\nAnswer:"

        # Call OpenRouter
        openai.ChatCompletion.create(
            model="openai/gpt-3.5-turbo",  # Or other via OpenRouter
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )


if __name__ == "__main__":
    chat_with_rag()
