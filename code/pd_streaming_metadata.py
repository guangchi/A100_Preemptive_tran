"""Extended NixlConnectorMetadata with push_chunks support.

Adds push_chunks to carry per-batch KV block deltas from the
P-side scheduler to the P-side worker, enabling stream_by_batch
(early KV transmission before request_finished).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds


def slice_block_ids(block_ids: BlockIds, offset: int, lengths: tuple[int, ...]) -> BlockIds:
    if offset < 0 or len(block_ids) != len(lengths):
        raise ValueError("invalid block slice geometry")
    result = tuple(group[offset:offset + length] for group, length in zip(block_ids, lengths))
    if any(len(group) != length for group, length in zip(result, lengths)):
        raise ValueError("block slice exceeds registered destination blocks")
    return result


def is_partial_completion_notif(notif: bytes) -> bool:
    return b"#chunk-" in notif


def chunk_ranges(start: int, end: int, max_blocks: int) -> list[tuple[int, int]]:
    if start < 0 or end < start or max_blocks <= 0:
        raise ValueError("invalid chunk range")
    return [(offset, min(offset + max_blocks, end))
            for offset in range(start, end, max_blocks)]


def has_write_capacity(max_in_flight: int, active_handles: int) -> bool:
    return max_in_flight <= 0 or active_handles < max_in_flight


@dataclass
class PushChunk:
    request_id: ReqId
    block_offset: int
    block_ids: BlockIds
    total_blocks: int
    is_final: bool
    ready_event: Any = field(default=None, repr=False, compare=False)
    prompt_tokens: int | None = None


class PreemptiveNixlMetadata(NixlConnectorMetadata):
    """NixlConnectorMetadata extended with partial (per-batch) push blocks."""

    def __init__(self) -> None:
        super().__init__()
        # Chunks are sent after the corresponding P forward completes.
        self.push_chunks: list[PushChunk] = []
