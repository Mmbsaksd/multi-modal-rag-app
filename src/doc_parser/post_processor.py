from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

@runtime_checkable
class ElementLike(Protocol):
    """Duck-typing protocol for parsed elements."""

    label: str
    text: str
    bbox: list[float]
    score: float
    reading_order: int
