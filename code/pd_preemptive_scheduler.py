"""vLLM 0.33 push scheduler: per-step block tables and post-forward chunks.

Long prompts use at most CHUNK_SIZE_BLOCKS per WRITE; short/medium prompts
aggregate each ready batch. The last prompt block is reserved for final delivery.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import (
    NixlPushConnectorScheduler,
)
from pd_streaming_metadata import PreemptiveNixlMetadata, PushChunk, chunk_ranges

logger = logging.getLogger("vllm." + __name__)
_STREAM_BY_BATCH = os.environ.get("PD_STREAM_BY_BATCH", "1") == "1"
_CHUNK_SIZE_BLOCKS = int(os.environ.get("CHUNK_SIZE_BLOCKS", "8"))


class PreemptivePushScheduler(NixlPushConnectorScheduler):
    def __init__(self, vllm_config, engine_id, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        if _CHUNK_SIZE_BLOCKS <= 0:
            raise ValueError("CHUNK_SIZE_BLOCKS must be positive")
        if self.use_host_buffer or len(kv_cache_config.transfer_groups) != 1:
            raise ValueError("Streaming currently requires one device KV cache group")
        self._active_requests: dict[str, Any] = {}
        self._active_block_ids: dict[str, tuple[list[int], ...]] = {}
        self._partial_cursor: dict[str, int] = {}
        logger.info("PreemptivePushScheduler: stream_by_batch=%s chunk_size=%d",
                    _STREAM_BY_BATCH, _CHUNK_SIZE_BLOCKS)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if (request.kv_transfer_params or {}).get("do_remote_decode"):
            self._active_requests[request.request_id] = request
            # Initial allocation only. Running-request additions come from
            # SchedulerOutput on every subsequent step, below.
            self._active_block_ids[request.request_id] = tuple(
                list(group) for group in blocks.get_unhashed_block_ids_all_groups()
            )

    def _emit(self, meta, request_id, blocks, start, end, total, final):
        chunk = tuple(list(group[start:end]) for group in blocks)
        if not all(len(group) == end - start for group in chunk):
            raise ValueError(f"incomplete block table for {request_id}: {start}:{end}")
        meta.push_chunks.append(PushChunk(request_id, start, chunk, total, final,
            prompt_tokens=self._active_requests[request_id].num_prompt_tokens))
        logger.info("PUSH_CHUNK_SCHEDULED req=%s offset=%d blocks=%d total=%d final=%d",
                    request_id, start, end - start, total, int(final))

    def build_connector_meta(self, scheduler_output):
        parent_meta = super().build_connector_meta(scheduler_output)
        meta = PreemptiveNixlMetadata()
        meta.__dict__.update(parent_meta.__dict__)
        # Snapshot values describe THIS forward, independent of later updates
        # to the live Request in the async scheduler.
        computed = {r.req_id: r.num_computed_tokens
                    for r in scheduler_output.scheduled_new_reqs}
        cached = scheduler_output.scheduled_cached_reqs
        computed.update(zip(cached.req_ids, cached.num_computed_tokens))
        for rid, block_groups, full_table in yield_req_data(scheduler_output):
            request = self._active_requests.get(rid)
            if request is None or rid in meta.push_finished_blocks:
                continue
            if block_groups is not None:
                if full_table:
                    if rid in cached.resumed_req_ids and self._partial_cursor.get(rid, 0):
                        raise RuntimeError("Streaming producer preemption requires restarting the remote request")
                    self._active_block_ids[rid] = tuple(list(g) for g in block_groups)
                else:
                    table = self._active_block_ids[rid]
                    self._active_block_ids[rid] = tuple(
                        list(old) + list(new) for old, new in zip(table, block_groups)
                    )
            if not _STREAM_BY_BATCH:
                continue
            blocks = self._active_block_ids[rid]
            prompt_tokens = request.num_prompt_tokens
            total = (prompt_tokens + self.block_size - 1) // self.block_size
            frontier_tokens = min(prompt_tokens, computed[rid]
                                  + scheduler_output.num_scheduled_tokens[rid])
            # Only full computed blocks. Reserve one prompt block so final is
            # always a nonempty WRITE carrying the sole completion notification.
            end = min(frontier_tokens // self.block_size, total - 1,
                      min(map(len, blocks)))
            cursor = self._partial_cursor.get(rid, 0)
            if end <= cursor:
                continue
            width = _CHUNK_SIZE_BLOCKS if prompt_tokens > 8192 else end - cursor
            for start, stop in chunk_ranges(cursor, end, width):
                self._emit(meta, rid, blocks, start, stop, total, False)
            self._partial_cursor[rid] = end

        for rid, blocks in list(meta.push_finished_blocks.items()):
            cursor = self._partial_cursor.get(rid, 0)
            total = max(map(len, blocks), default=0)
            width = _CHUNK_SIZE_BLOCKS if total * self.block_size > 8192 and _STREAM_BY_BATCH else max(1, total)
            for start, end in chunk_ranges(cursor, total, width):
                self._emit(meta, rid, blocks, start, end, total, end == total)
            del meta.push_finished_blocks[rid]
            self._forget(rid)
        for rid in meta.reqs_not_processed:
            self._forget(rid)
        return meta

    def _forget(self, rid):
        self._partial_cursor.pop(rid, None)
        self._active_requests.pop(rid, None)
        self._active_block_ids.pop(rid, None)
