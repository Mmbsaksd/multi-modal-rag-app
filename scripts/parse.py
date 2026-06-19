"""CLI entry point for document parsing pipeline.

Usage:
    python scripts/parse.py document.pdf
    python scripts/parse.py ./docs/ --output ./parsed/ --format markdown
    python scripts/parse.py document.pdf --chunks --output ./chunks/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path(0, str(Path(__file__).parent.parent/"src"))

from rich.console import Console
from rich.logging import RichHandler