import os
from typing import Optional

import openai
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class ClientConfig(BaseSettings):
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-3.5-turbo"
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    @model_validator(mode="after")
    def set_provider_defaults(self):
        if self.provider == "openrouter":
            self.api_base = self.api_base or "https://openrouter.ai/api/v1"
            self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        else:
            self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(f"API key required for provider {self.provider}")
        return self

    def __init__(self, **data):
        super().__init__(**data)
        if self.provider == "openrouter":
            self.api_base = self.api_base or "https://openrouter.ai/api/v1"
            self.api_key = os.getenv("OPENROUTER_API_KEY", self.api_key)
            # Map model if needed, e.g., self.model = "openai/gpt-3.5-turbo"


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
    config = get_client_config("openai")
    config.temperature = 0.0  # Deterministic for retrieval
    return create_client(config)
