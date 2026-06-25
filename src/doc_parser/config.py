from __future__ import annotations

import logging

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    # Parser backend
    parser_backend: str = "cloud"  # "cloud" | "ollama"
    z_ai_api_key: SecretStr | None = None
    log_level: str = "INFO"
    output_dir: str = "./output"
    config_yaml_path: str = "config.yaml"

    azure_openai_api_key: SecretStr
    azure_openai_endpoint: str
    azure_openai_api_version: str = "2025-01-01-preview"

    embedding_provider: str = "azure"  # "azure" | "gemini"
    azure_openai_embedding_deployment: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    gemini_api_key: SecretStr | None = None


    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection_name: str = "documents"

    reranker_backend: str = "openai"  # "jina" | "openai" | "bge" | "qwen"
    reranker_top_n: int = 5
    jina_api_key: SecretStr | None = None








def get_settings() -> Settings:
    """Return the singleton Settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings
    return _settings

def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with the given level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )