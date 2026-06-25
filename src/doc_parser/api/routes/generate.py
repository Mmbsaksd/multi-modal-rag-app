from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException
from loguru import logger
from doc_parser.api.dependencies import (
    get
)