from __future__ import annotations

import logging
from dataclasses import dataclass, field

from doc_parser.post_processor import ElementLike

logger = logging.getLogger(__name__)

_TOKEN_WORD_RATIO: float = 1.3

ATOMIC_LABELS: frozenset[str] = frozenset(
    {"table", "formula", "inline_formula", "algorithm", "image", "figure"}
)

TITLE_LABELS: frozenset[str] = frozenset(
    {"document_title", "paragraph_title", "figure_title"}
)

# Modality classification sets
_IMAGE_TYPES: frozenset[str] = frozenset({"image", "figure"})
_TABLE_TYPES: frozenset[str] = frozenset({"table"})
_FORMULA_TYPES: frozenset[str] = frozenset({"formula", "inline_formula"})
_ALGORITHM_TYPES: frozenset[str] = frozenset({"algorithm"})

def _infer_modality(element_types: list[str]) -> str:
    """Derive chunk modality from element label(s).

    Args:
        element_types: List of element labels in the chunk.

    Returns:
        One of: "image", "table", "formula", "algorithm", "text".
    """
    types = frozenset(element_types)
    if types & _IMAGE_TYPES:
        return "image"
    if types & _TABLE_TYPES:
        return "table"
    if types & _FORMULA_TYPES:
        return "formula"
    if types & _ALGORITHM_TYPES:
        return "algorithm"
    return "text"


@dataclass
class Chunk:
    text: str
    chunk_id: str
    page: int
    element_types: list[str]
    bbox: list[float] | None
    source_file: str
    is_atomic: bool
    modality: str = field(default="text")
    image_base64: str | None = field(default=None)
    caption: str | None = field(default=None)

def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _TOKEN_WORD_RATIO)

def _split_text_into_sub_chunks(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    words_per_chunk = max(1, int(max_tokens / _TOKEN_WORD_RATIO))
    sub_chunks = []
    for i in range(0, len(words), words_per_chunk):
        sub_chunks.append(" ".join(words[i : i + words_per_chunk]))
    return sub_chunks

def document_aware_chunking(
    pages: list[tuple[int, list[ElementLike]]],
    source_file: str,
    max_chunk_tokens: int = 512,
):
    all_pairs: list[tuple[int, ElementLike]] = [
        (page_num, el)
        for page_num, elements in pages
        for el in elements
    ]
    if not all_pairs:
        return []
    
    all_pairs.sort(key=lambda x: (x[0], x[1].reading_order))
    chunks: list[Chunk] = []
    chunk_idx = 0

    current_texts: list[str] = []
    current_labels: list[str] = []
    current_tokens: int = 0
    current_page: int = all_pairs[0][0] 

    pending_title: str | None = None
    pending_title_label: str | None = None
    pending_title_page: int = current_page

    def flush_current()-> None:
        nonlocal current_texts, current_labels, current_tokens, chunk_idx, chunks
        nonlocal pending_title, pending_title_label, pending_title_page, current_page



        if not current_texts and pending_title is None:
            return
        


        texts_to_flush: list[str] = []
        labels_to_flush: list[str] = []

        page_to_use = pending_title_page if (pending_title and not current_texts) else current_page


        if pending_title is not None:
            texts_to_flush.append(pending_title)
            labels_to_flush.append(pending_title_label or "paragraph_title")
            pending_title = None
            pending_title_label = None

        texts_to_flush.extend(current_texts)
        labels_to_flush.extend(current_labels)

        if not texts_to_flush:
            return
        
        chunk = Chunk(
            text="\n\n".join(texts_to_flush),
            chunk_id=f"{source_file}_{page_to_use}_{chunk_idx}",
            page=page_to_use,
            element_types=labels_to_flush,
            bbox=None,
            source_file=source_file,
            is_atomic=False,
            modality=_infer_modality(labels_to_flush),
        )
        chunks.append(chunk)
        chunk_idx += 1
        current_texts = []
        current_labels = []
        current_tokens = 0

    for page_num, element in all_pairs:
        label = element.label
        text = element.text.strip()

        if label in ATOMIC_LABELS:
            figure_caption: str | None = None
            if pending_title is not None and pending_title_label == "figure_title":
                figure_caption = pending_title
                pending_title = None
                pending_title_label = None
            
            flush_current()

            if figure_caption:
                atomic_text = f"{figure_caption}\n\n{text}" if text else figure_caption
                atomic_labels = ["figure_title", label]
            else:
                atomic_text = text
                atomic_labels = [label]

            atomic_chunk = Chunk(
                text=atomic_text,
                chunk_id=f"{source_file}_{page_num}_{chunk_idx}",
                page=page_num,
                element_types=atomic_labels,
                bbox=element.bbox,
                source_file=source_file,
                is_atomic=True,
                modality=_infer_modality(atomic_labels),
            )

            chunks.append(atomic_chunk)
            chunk_idx += 1
            continue

        if not text:
            continue

        if label in TITLE_LABELS:
            if current_texts:
                flush_current()
            elif pending_title is not None:
                flush_current()

            pending_title = text

        token_estimate = _estimate_tokens(text)
        pending_token = _estimate_tokens(pending_title) if pending_title else 0

        if token_estimate > max_chunk_tokens:
            flush_current()
            sub_chunks = _split_text_into_sub_chunks(text, max_chunk_tokens)
            for sub_text in sub_chunks:
                chunk = Chunk(
                    text=sub_text,
                    chunk_id=f"{source_file}_{page_num}_{chunk_idx}",
                    page=page_num,
                    element_types=[label],
                    bbox=None,
                    source_file=source_file,
                    is_atomic=False,
                    modality=_infer_modality([label]),
                )
                chunks.append(chunk)
                chunk_idx += 1
            continue

        if current_texts and (current_tokens + token_estimate + pending_token > max_chunk_tokens):
            flush_current()

        if pending_title is not None:
            if not current_texts:
                current_page = pending_title_page  # chunk's page = where heading started
            current_texts.append(pending_title)
            current_labels.append(pending_title_label or "paragraph_title")
            current_tokens += _estimate_tokens(pending_title)
            pending_title = None
            pending_title_label = None

        if not current_texts:
            current_page = page_num  # first element sets the chunk's page

        current_texts.append(text)
        current_labels.append(label)
        current_tokens += token_estimate

        if current_tokens >= max_chunk_tokens:
            flush_current()

    flush_current()
    return chunks