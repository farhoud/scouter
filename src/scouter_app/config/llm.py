import os
from functools import lru_cache

import openai
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.llm import OpenAILLM
from pydantic import model_validator
from pydantic_settings import BaseSettings


class ClientConfig(BaseSettings):
    provider: str = "openai"
    api_key: str | None = None
    model: str = "x-ai/grok-4.1-fast:free"
    api_base: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    env: str = "test"

    @model_validator(mode="after")
    def set_provider_defaults(self):
        if self.provider == "openrouter":
            self.api_base = self.api_base or "https://openrouter.ai/api/v1"
            self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        else:
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(f"API key required for provider {self.provider}")
        if self.env not in ["development", "production", "test"]:
            raise ValueError("env must be one of: development, production, test")
        return self

    def __init__(self, **data):
        super().__init__(**data)
        if self.provider == "openrouter":
            self.api_base = self.api_base or "https://openrouter.ai/api/v1"
            self.api_key = os.getenv("OPENROUTER_API_KEY", self.api_key)
            # Map model if needed, e.g., self.model = "openai/gpt-3.5-turbo"
        self.env = os.getenv("SCOUTER_ENV", self.env)


def get_client_config(provider: str = "openai") -> ClientConfig:
    return ClientConfig(provider=provider)


def create_client(config: ClientConfig) -> openai.OpenAI:
    return openai.OpenAI(
        api_key=config.api_key,
        base_url=config.api_base,
    )


def get_chatbot_client() -> openai.OpenAI:
    config = get_client_config("openrouter")
    config.temperature = 0.9  # More creative for chatbot
    return create_client(config)


def get_scouter_client() -> openai.OpenAI:
    config = get_client_config("openrouter")
    config.temperature = 0.0  # Deterministic for retrieval
    return create_client(config)


@lru_cache(maxsize=1)
def get_neo4j_llm() -> OpenAILLM:
    config = get_client_config("openrouter")
    return OpenAILLM(config.model, api_key=config.api_key, base_url=config.api_base)


@lru_cache(maxsize=1)
def get_neo4j_embedder() -> SentenceTransformerEmbeddings:
    return SentenceTransformerEmbeddings("Qwen/Qwen3-Embedding-0.6B")
