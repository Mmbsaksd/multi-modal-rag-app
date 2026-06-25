from __future__ import annotations
from functools import lru_cache

from openai import AsyncAzureOpenAI

from doc_parser.config import Settings, get_settings
from doc_parser.ingestion.embedder import BaseEmbedder, get_embedder
from doc_parser.ingestion.vector_store import QdrantDocumentStore
from doc_parser.retrieval.reranker import BaseReranker, get_reranker


@lru_cache
def get_azureopenai_client()-> AsyncAzureOpenAI:
    settings = get_settings()
    return AsyncAzureOpenAI(
        api_key=settings.azure_openai_api_key.get_secret_value(),
        azure_endpoint=settings.azure_openai_endpoint,
        api_version=settings.azure_openai_api_version
    )

@lru_cache
def get_store() -> QdrantDocumentStore:
    """Return a cached QdrantDocumentStore."""
    return QdrantDocumentStore(get_settings())

@lru_cache
def get_reranker_dep() -> BaseReranker:
    """Return a cached BaseReranker for the configured backend."""
    return get_reranker(get_settings())

@lru_cache
def get_embedder_dep() -> BaseEmbedder:
    """Return a cached BaseEmbedder for the configured provider."""
    return get_embedder(get_settings())
