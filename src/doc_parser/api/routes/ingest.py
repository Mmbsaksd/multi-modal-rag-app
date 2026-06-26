"""POST /ingest endpoint — file upload and JSON-path variants."""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from loguru import logger
from qdrant_client.models import SparseVector

from doc_parser.api.dependencies import get_embedder_dep, get_azureopenai_client, get_store
from doc_parser.api.schemas import IngestRequest, IngestResponse
from doc_parser.chunker import Chunk, document_aware_chunking
from doc_parser.config import get_settings
from doc_parser.ingestion.embedder import embed_chunks


























