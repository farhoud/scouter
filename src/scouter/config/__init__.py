from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass


@dataclass
class LLMConfig:
    provider: str = "openai"
    api_key: str | None = None
    model: str = "openai/gpt-oss-20b:free"
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int = 30
    max_retries: int = 3
    env: str = "test"

    @classmethod
    def load_from_env(cls) -> LLMConfig:
        provider = os.getenv("LLM_PROVIDER", "openai")
        if provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
        else:
            # Default to openai
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            key_name = (
                "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
            )
            msg = f"API key required for provider '{provider}'. Set {key_name} environment variable."
            raise ValueError(msg)

        env = os.getenv("ENV", "test")
        if env not in ["development", "production", "test"]:
            msg = "env must be one of: development, production, test"
            raise ValueError(msg)

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            env=env,
        )


@dataclass
class DBConfig:
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    embedder_model: str = "Qwen/Qwen3-Embedding-0.6B"
    llm_model: str = "openai/gpt-oss-20b:free"

    @classmethod
    def load_from_env(cls) -> DBConfig:
        return cls(
            uri=os.getenv("NEO4J_URI", cls.uri),
            user=os.getenv("NEO4J_USER", cls.user),
            password=os.getenv("NEO4J_PASSWORD", cls.password),
        )


@dataclass
class LoggingConfig:
    level: str = "INFO"


@dataclass
class AppConfig:
    llm: LLMConfig
    db: DBConfig
    logging: LoggingConfig

    @classmethod
    def load_from_env(cls) -> AppConfig:
        return cls(
            llm=LLMConfig.load_from_env(),
            db=DBConfig.load_from_env(),
            logging=LoggingConfig(),
        )


config = AppConfig.load_from_env()


def setup_logging(level: str | None = None) -> None:
    """Setup logging for the application."""
    level = level or config.logging.level
    logger = logging.getLogger("scouter")
    logger.setLevel(getattr(logging, level.upper()))

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    logger.propagate = False
