"""Re-ranker backends for post-retrieval relevance scoring.

Pipeline position:
    QdrantDocumentStore.search() → top-k candidates
        ↓
    BaseReranker.rerank(query, candidates)  ← this module
        ↓
    LLM generation

Supported backends (controlled by ``RERANKER_BACKEND`` env var):
    - ``openai``  – GPT-4o-mini as async cross-encoder (default, no extra deps)
    - ``jina``    – Jina Reranker M0 cloud API (multimodal, needs JINA_API_KEY)
    - ``bge``     – BAAI/bge-reranker-v2-minicpm-layerwise (local, fast, text-only)
    - ``qwen``    – Qwen3-VL-Reranker-2B (local, multimodal, heavier)
"""
from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import httpx
from openai import AsyncAzureOpenAI

if TYPE_CHECKING:
    from doc_parser.config import Settings

logger = logging.getLogger(__name__)

class BaseReranker(ABC):
    """Abstract re-ranker interface.

    All concrete backends must implement :meth:`rerank`.
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Re-rank *candidates* against *query*, returning the top-n most relevant.

        Args:
            query: The user's natural-language query.
            candidates: Payload dicts returned by ``QdrantDocumentStore.search()``.
                Each dict contains at minimum ``"text"``, ``"modality"``, and
                optionally ``"image_base64"`` for image chunks.
            top_n: Maximum number of results to return (highest score first).

        Returns:
            Up to *top_n* candidate dicts, sorted by relevance (best first).
            Each dict is the original payload extended with a ``"rerank_score"`` key.
        """


class AzureOpenAIReranker(BaseReranker):
    """Re-rank using GPT-4o-mini as an async cross-encoder.

    Scores each (query, chunk) pair via a short prompt, firing all candidates
    in parallel with ``asyncio.gather``.  Image chunks pass ``image_base64``
    inline as a vision message.  Text-only chunks use a text-only message.

    Cost: ~$0.03–0.10 per re-rank call (20 candidates).
    Latency: ~800ms–2s (parallel async).
    """

    _SCORE_PROMPT = (
        "Rate the relevance of the following document to the query on a scale of 1 to 10. "
        "Reply with ONLY the integer score (e.g. '7'), nothing else.\n\n"
        "Query: {query}\n\nDocument: {text}"
    )
    def __init__(self, settings: "Settings") -> None:
        api_key = (
            settings.azure_openai_api_key.get_secret_value()
            if settings.azure_openai_api_key
            else None
        )
        self._client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version
        )
        self._model = "gpt-4o-mini"

    async def _score_one(self, query: str, candidate: dict[str, Any]) -> float:
        """Return a relevance score in [1, 10] for one candidate."""
        text = candidate.get("text") or ""
        image_b64 = candidate.get("image_base64")
        modality = candidate.get("modality", "text")
        if modality == "image" and image_b64:
            messages: list[dict[str,Any]] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Rate the relevance of the following image (and its caption) "
                                f"to the query on a scale of 1 to 10. "
                                f"Reply with ONLY the integer score.\n\nQuery: {query}"
                                + (f"\n\nCaption: {text}" if text else "")
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ]
        else:
            prompt = self._SCORE_PROMPT.format(query=query, text=text[:2000])
            messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                max_tokens=4,
            )
            raw = (response.choices[0].message.content or "").strip()
            return float(raw)
        except (ValueError, IndexError):
            logger.warning("Could not parse score from AzureOpenAI response: %r", raw)
            return 0.0
        except Exception as exc:
            logger.error("OpenAI scoring failed for chunk: %s", exc)
            return 0.0











