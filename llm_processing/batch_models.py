"""Data models for batch processing."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchItem:
    """A token-safe batch of rows with metadata."""

    batch_id: int
    batch_rows: int
    batch_tokens: int
    row_separator: str
    id_list: list[str]
    rows: list[str]


@dataclass
class BatchProcessResult:
    """Result from processing a single batch."""

    batch_rows: int
    id_list: list[str]
    batch_result: Any
    error_type: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    batch_item: BatchItem | None = None
