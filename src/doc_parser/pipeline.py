"""Main document parsing pipeline wrapping the GLM-OCR SDK."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from doc_parser.config import get_settings
from doc_parser.post_processor import assemble_markdown, save_to_json
from uti