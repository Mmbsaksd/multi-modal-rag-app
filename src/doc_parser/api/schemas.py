"""Pydantic request/response models for the doc-parser RAG API."""
from __future__ import annotations

from pydantic import BaseModel, Field
# ── Request models ─────────────────────────────────────────────────────────────








# ── Response models ────────────────────────────────────────────────────────────

class ChunkResult(BaseModel):
    """A single retrieved and (optionally) reranked document chunk."""

    chunk_id: str
    text: str
    source_file: str
    page: int
    modality: str
    element_types: list[str]
    bbox: list[float] | None
    is_atomic: bool
    caption: str | None
    rerank_score: float | None
    image_base64: str | None = None



class GenerateRequest(BaseModel):
    """Request body for POST /generate."""

    query: str = Field(..., description="Natural-language question to answer.")
    top_k: int = Field(20, ge=1, le=200, description="Candidate count from Qdrant.")
    top_n: int | None = Field(None, ge=1, description="Context chunks after reranking.")
    filter_modality: str | None = Field(
        None, description='"text"|"image"|"table"|"formula" or null for all.'
    )
    rerank: bool = Field(True, description="If False, use raw Qdrant results.")
    system_prompt: str | None = Field(None, description="Override default RAG system prompt.")
    max_tokens: int = Field(1024, ge=64, le=4096, description="Max tokens in LLM response.")

class GenerateResponse(BaseModel):
    """Response body for POST /generate."""

    query: str
    answer: str
    sources: list[ChunkResult]
    total_candidates: int
    latency_ms: float





