import os

import requests
from scouter_app.config.llm import get_chatbot_client

HTTP_OK = 200

SCOUTER_SEARCH_URL = "http://localhost:8000/v1/search"

# Get LLM client
llm = get_chatbot_client()


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

        # Call LLM
        response = llm.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # Or other via OpenRouter
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        answer = response.choices[0].message.content
        print(f"Bot: {answer}")


if __name__ == "__main__":
    chat_with_rag()
