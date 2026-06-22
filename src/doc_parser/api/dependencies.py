from __future__ import annotations
from functools import lru_cache

from openai import AsyncAzureOpenAI

from doc_parser.config import Settings, get_settings
from doc_parser.ingestion