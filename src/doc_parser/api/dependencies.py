from __future__ import annotations
from functools import lru_cache

from openai import AsyncAzureOpenAI

from doc_parser.config import Settings, get_settings
from doc_parser.ingestion.embedder import BaseEmbedder, get_embedder
from doc_parser.ingestion.vector_store import QdrantDocumentStore
from doc_parser.re