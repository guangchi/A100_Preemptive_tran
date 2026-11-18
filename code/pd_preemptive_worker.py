"""P-side push worker with preemptive KV transfer.

Extends NixlPushConnectorWorker with:
  1. stream_by_batch: reads push_chunks from metadata and delivers
     partial KV block chunks to the writer thread ahead of request_finished.
  2. preemptive_transfer: writer thread uses a priority-aware queue where
     short requests can bypass long requests at chunk boundaries, subject to
     three anti-starvation protections on long requests.

Request categories (by total_blocks across all KV groups):
  short  : total_blocks <= SHORT_BLOCKS   (prompt < 256 tokens)
  medium : SHORT_BLOCKS < total_blocks <= LONG_BLOCKS (256 <= prompt <= 8192)
  long   : total_blocks > LONG_BLOCKS     (prompt > 8192 tokens)

Three-protection rule (long requests are protected when ANY holds):
  P1 progress : success_blocks / total_blocks >= PROTECT_PROGRESS (0.85)
  P2 bypass   : bypass_count >= PROTECT_BYPASS   (3 short chunks bypassed)
  P3 wait     : bypass_wait_ms >= PROTECT_WAIT_MS (50 ms cumulative wait)

Once protected, the long request is selected next regardless of queue order,
until its current chunk WRITE completes.

Environment variables:
  PD_PRIORITY_PREEMPTION   1=enabled (default), 0=disabled (pure FIFO)
  LONG_REQUEST_PROTECT_PROGRESS   float  default 0.85
  LONG_REQUEST_MAX_PREEMPTIONS    int    default 3
  LONG_REQUEST_MAX_PREEMPT_WAIT_MS float default 50.0
  PD_STREAM_MAX_IN_FLIGHT  int    max concurrent NIXL WRITE handles (default 1)
"""
from __future__ import annotations

import os
import queue
import time
import logging
from pd_pipeline_trace import emit
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
    NixlPushConnectorWorker,
    PUSH_REG_NOTIF_PREFIX,
)
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds

from pd_streaming_metadata import (
    PreemptiveNixlMetadata,
    has_write_capacity,
    is_partial_completion_notif,
    slice_block_ids,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import get_base_request_id

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

# Attach to the "vllm" logger tree so vLLM's configured handler emits these
# records.  A standalone logger (logging.getLogger(__name__)) has no handler
# and does not propagate into the vllm tree, so its output is silently lost.
logger = logging.getLogger("vllm." + __name__)

# ── Thresholds ────────────────────────────────────────────────────────────── #
# block_size=16 tokens/block → 256 tokens = 16 blocks, 8192 tokens = 512 blocks
_SHORT_BLOCKS: int = 16
_LONG_BLOCKS: int = 512

@dataclass
class LongProtection:
    protect_progress: float
    protect_bypass: int
    protect_wait_ms: float

LONG_PROTECT = LongProtection(
    protect_progress=float(os.environ.get("LONG_REQUEST_PROTECT_PROGRESS", "0.85")),
    protect_bypass=int(os.environ.get("LONG_REQUEST_MAX_PREEMPTIONS", "3")),
    protect_wait_ms=float(os.environ.get("LONG_REQUEST_MAX_PREEMPT_WAIT_MS", "50.0")),
)

_PRIORITY_PREEMPTION: bool = os.environ.get("PD_PRIORITY_PREEMPTION", "1") == "1"
_MAX_IN_FLIGHT: int = int(os.environ.get("PD_STREAM_MAX_IN_FLIGHT", "1"))
_POLL_MS: float = float(os.environ.get("PD_STREAM_POLL_MS", "0.1"))
_TRACE_PIPELINE = bool(os.environ.get("PD_PIPELINE_TRACE"))


def classify_prompt_length(total_blocks: int) -> str:
    if total_blocks <= _SHORT_BLOCKS:
        return "short"
    elif total_blocks <= _LONG_BLOCKS:
        return "medium"
    return "long"


# ── Per-request protection state ──────────────────────────────────────────── #

@dataclass
class ReqProtectionState:
    req_id: str
    total_blocks: int
    success_blocks: int = 0
    bypass_count: int = 0
    bypass_wait_start: float = 0.0
    bypass_wait_ms: float = 0.0
    is_protected: bool = False

    def category(self) -> str:
        return classify_prompt_length(self.total_blocks)

    def refresh_wait(self) -> None:
        if self.bypass_wait_start > 0:
            now = time.perf_counter()
            self.bypass_wait_ms += (now - self.bypass_wait_start) * 1000
            self.bypass_wait_start = now

    def update_protection(self) -> None:
        if self.is_protected:
            return
        progress = self.success_blocks / max(1, self.total_blocks)
        if (
            progress >= LONG_PROTECT.protect_progress
            or self.bypass_count >= LONG_PROTECT.protect_bypass
            or self.bypass_wait_ms >= LONG_PROTECT.protect_wait_ms - 1e-9
        ):
            self.is_protected = True
            logger.info(
                "Long request %s is now protected: progress=%.2f "
                "bypass=%d wait_ms=%.1f",
                self.req_id, progress, self.bypass_count, self.bypass_wait_ms,
            )

    def record_success(self, n_blocks: int) -> None:
        self.success_blocks += n_blocks
        self.update_protection()

    def record_bypass_start(self) -> None:
        """Called when a short chunk jumps ahead of this long request."""
        self.bypass_count += 1
        if self.bypass_wait_start == 0.0:
            self.bypass_wait_start = time.perf_counter()

    def record_bypass_resume(self) -> None:
        """Called when this long request is selected again after being bypassed."""
        if self.bypass_wait_start > 0.0:
            self.bypass_wait_ms += (time.perf_counter() - self.bypass_wait_start) * 1000
            self.bypass_wait_start = 0.0
        self.update_protection()


# ── Priority-aware transfer queue item ───────────────────────────────────── #

@dataclass(order=False)
class TransferItem:
    req_id: str
    block_chunk: BlockIds
    total_blocks: int   # 0 = unknown (use len(block_chunk))
    is_final: bool      # True = comes from push_finished_blocks (request done)
    block_offset: int = 0
    ready_event: Any = None
    inserted_at: float = field(default_factory=time.perf_counter)
    submitted_at: float = 0.0
    prompt_tokens: int | None = None

    def effective_total(self) -> int:
        return self.total_blocks if self.total_blocks > 0 else sum(
            len(g) for g in self.block_chunk
        )

    def category(self) -> str:
        if self.prompt_tokens is not None:
            return 'short' if self.prompt_tokens < 256 else 'medium' if self.prompt_tokens <= 8192 else 'long'
        return classify_prompt_length(self.effective_total())


# ── Worker ───────────────────────────────────────────────────────────────── #

class PreemptivePushWorker(NixlPushConnectorWorker):
    """Writer advances handles; engine thread owns request/lease completion.

    Experimental streaming is restricted to the TP=PP=1 device-cache setup
    used by this study. Each chunk maps to one WRITE handle.
    """

    def __init__(self, vllm_config, engine_id, kv_cache_config):
        pc = vllm_config.parallel_config
        if pc.tensor_parallel_size != 1 or pc.pipeline_parallel_size != 1:
            raise ValueError("Streaming WRITE limits currently require TP=PP=1")
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._transfer_queue: list[TransferItem] = []
        self._stream_inbox = queue.Queue()
        self._chunk_complete_inbox = queue.Queue()
        self._prot_state: dict[str, ReqProtectionState] = {}
        self._stream_registrations: dict[str, dict[str, Any]] = {}
        self._next_offset: dict[str, int] = {}
        self._busy_requests: set[str] = set()
        self._failed_streams: set[str] = set()
        # Writer owns handle bookkeeping; sending lock protects final handoff.
        self._chunk_transfers: dict[str, TransferItem] = {}
        self._stream_done_transfers: set[str] = set()
        self._stream_failed_transfers: set[str] = set()
        logger.info("PreemptivePushWorker: preemption=%s max_in_flight=%d protect=%s",
                    _PRIORITY_PREEMPTION, _MAX_IN_FLIGHT, LONG_PROTECT)

    def start_load_kv(self, metadata, **kwargs):
        if metadata.push_finished_blocks:
            raise RuntimeError("Streaming scheduler must convert final blocks to chunks")
        super().start_load_kv(metadata, **kwargs)

    def mark_stream_chunks_ready(self, metadata):
        if not isinstance(metadata, PreemptiveNixlMetadata) or not metadata.push_chunks:
            return
        import torch
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        for chunk in metadata.push_chunks:
            if chunk.block_offset == 0 or chunk.is_final:
                emit("kv_frontier_recorded", rid=chunk.request_id, offset=chunk.block_offset,
                     final=chunk.is_final)
            self._stream_inbox.put(TransferItem(
                req_id=chunk.request_id, block_chunk=chunk.block_ids,
                total_blocks=chunk.total_blocks, is_final=chunk.is_final,
                block_offset=chunk.block_offset, ready_event=event,
                prompt_tokens=chunk.prompt_tokens,
            ))
        self._push_writer_wake.set()

    def _pop_done_transfers(self, transfers):
        if transfers is not self._sending_transfers:
            return super()._pop_done_transfers(transfers)
        # Engine holds the sending lock. Only the writer touches live WRITE
        # handles, including the gap between submission and submission logging.
        done = set(self._stream_done_transfers)
        failed = set(self._stream_failed_transfers)
        self._stream_done_transfers.clear()
        self._stream_failed_transfers.clear()
        return done, failed

    def _record_chunk_completions(self, done, failed):
        """Called by the writer under the sending lock."""
        for tid in done | failed:
            item = self._chunk_transfers.get(tid)
            if item is not None:
                failure = tid in failed or tid in self._send_failures
                self._chunk_complete_inbox.put((item, failure, tid in done))
                if tid in done:
                    self._chunk_transfers.pop(tid, None)
                    if not item.is_final:
                        # Partial IDs are internal, never consumed by the
                        # parent's request-level failure/lease bookkeeping.
                        self._send_failures.discard(tid)
        if done or failed:
            self._push_writer_wake.set()

    def _poll_stream_transfers(self):
        """Advance chunks without waiting for another model/engine step.

        The parent lock protects publication and completion-result handoff.
        Final IDs are retained for the engine's normal lease accounting.
        """
        with self._sending_transfers_lock:
            if not self._sending_transfers:
                return
            done, failed = super()._pop_done_transfers(self._sending_transfers)
            self._send_failures.update(failed)
            self._record_chunk_completions(done, failed)
            self._stream_done_transfers.update(tid for tid in done if "#chunk-" not in tid)
            self._stream_failed_transfers.update(tid for tid in failed if "#chunk-" not in tid)

    def _drain_inbox(self, inbox):
        while True:
            try:
                yield inbox.get_nowait()
            except queue.Empty:
                return

    def _push_writer_loop(self):
        import msgspec
        while not self._push_writer_stop.is_set():
            self._push_writer_wake.clear()
            try:
                self._poll_stream_transfers()
                for rid, rd in self._drain_inbox(self._reg_send_inbox):
                    self._send_registration_to_p(rid, rd)
                for item in self._drain_inbox(self._stream_inbox):
                    self._transfer_queue.append(item)
                    if item.category() == "long":
                        self._prot_state.setdefault(item.req_id, ReqProtectionState(
                            req_id=item.req_id, total_blocks=item.total_blocks))
                for item, failed, done in self._drain_inbox(self._chunk_complete_inbox):
                    if failed:
                        self._failed_streams.add(item.req_id)
                    if done:
                        self._busy_requests.discard(item.req_id)
                        if not failed and item.req_id not in self._failed_streams:
                            count = max(map(len, item.block_chunk), default=0)
                            self._next_offset[item.req_id] = item.block_offset + count
                            state = self._prot_state.get(item.req_id)
                            if state:
                                state.record_success(count)
                            if item.block_offset == 0 or item.is_final:
                                emit("write_done_observed", rid=item.req_id, offset=item.block_offset,
                                     final=item.is_final)
                            logger.info("PUSH_CHUNK_DONE req=%s offset=%d blocks=%d final=%d observed_ms=%.3f",
                                        item.req_id, item.block_offset, count, int(item.is_final),
                                        (time.perf_counter() - item.submitted_at) * 1000)
                for rid in self._drain_inbox(self._evict_finished_inbox):
                    self._transfer_queue = [it for it in self._transfer_queue if it.req_id != rid]
                    self._stream_registrations.pop(get_base_request_id(rid), None)
                    self._next_offset.pop(rid, None)
                    self._prot_state.pop(rid, None)
                    self._failed_streams.discard(rid)
                    # A still-active failed/expired WRITE keeps the request busy
                    # until the base worker can release its DMA handle.

                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            reg = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX):])
                            rid = reg.get("request_id") if isinstance(reg, dict) else None
                            if not isinstance(rid, str):
                                raise ValueError("invalid PUSH_REG request ID")
                            self._stream_registrations[get_base_request_id(rid)] = reg
                            logger.info("PUSH_REGISTERED req=%s blocks=%d", rid,
                                        max(map(len, reg['local_block_ids']), default=0))
                        elif is_partial_completion_notif(notif):
                            logger.debug("PUSH_PARTIAL_NOTIFICATION_IGNORED %s", notif)
                        else:
                            if not notif.startswith(b"HB:"):
                                logger.info("PUSH_FINAL_NOTIFICATION %s", notif)
                            self._pending_completion_notifs.put(notif)
                self._dispatch_from_priority_queue()
            except Exception:
                logger.exception("preemptive-push-writer error; request retained for lease cleanup")
            # Registrations may arrive while P is idle, so keep polling whenever
            # there is outstanding queue/registration/WRITE state.
            if self._transfer_queue or self._stream_registrations or self._busy_requests:
                self._push_writer_wake.wait(timeout=_POLL_MS / 1000)
            else:
                self._push_writer_wake.wait(timeout=0.1)

    def _has_write_capacity(self):
        with self._sending_transfers_lock:
            count = sum(map(len, self._sending_transfers.values()))
        return has_write_capacity(_MAX_IN_FLIGHT, count)

    def _eligible(self, item):
        if item.req_id in self._failed_streams or item.req_id in self._busy_requests:
            return False
        if item.block_offset != self._next_offset.get(item.req_id, 0):
            return False
        reg = self._stream_registrations.get(get_base_request_id(item.req_id))
        if reg is None or (item.ready_event is not None and not item.ready_event.query()):
            return False
        if _TRACE_PIPELINE and (item.block_offset == 0 or item.is_final) and not getattr(item, '_trace_ready', False):
            item._trace_ready = True
            # Observation during normal arbitration, not the exact CUDA
            # completion instant. No extra event queries are introduced.
            emit("chunk_ready_observed", rid=item.req_id, offset=item.block_offset,
                 final=item.is_final)
        fut = self._ensure_handshake(reg['decode_engine_id'], reg['decode_host'],
                                     reg['decode_port'], reg['decode_tp_size'])
        if fut is not None:
            if fut.done() and fut.exception() is not None:
                self._failed_streams.add(item.req_id)
                logger.error("PUSH handshake failed req=%s: %s", item.req_id, fut.exception())
            return False
        if _TRACE_PIPELINE and (item.block_offset == 0 or item.is_final) and not getattr(item, '_trace_eligible', False):
            item._trace_eligible = True
            emit("chunk_eligible_observed", rid=item.req_id, offset=item.block_offset,
                 final=item.is_final)
        return True

    def _select_next_item(self):
        candidates = [(i, it) for i, it in enumerate(self._transfer_queue) if self._eligible(it)]
        if not candidates:
            return None
        for state in self._prot_state.values():
            state.refresh_wait()
            state.update_protection()
        selected = candidates[0]
        if _PRIORITY_PREEMPTION:
            protected = [(i, it) for i, it in candidates
                         if (state := self._prot_state.get(it.req_id)) and state.is_protected]
            shorts = [(i, it) for i, it in candidates if it.category() == 'short']
            if protected:
                selected = protected[0]
            elif shorts and selected[1].category() == 'long':
                selected = shorts[0]
                for i, long_item in candidates:
                    if i >= selected[0]:
                        break
                    state = self._prot_state.get(long_item.req_id)
                    if state:
                        state.record_bypass_start()
                        state.update_protection()
                        logger.info("Preemption: short req %s bypasses long req %s",
                                    selected[1].req_id, long_item.req_id)
        if _TRACE_PIPELINE and any(it.category() == 'short' for _, it in candidates):
            emit("writer_choice", rid=selected[1].req_id, offset=selected[1].block_offset,
                 final=selected[1].is_final, selected_index=selected[0],
                 eligible=[dict(rid=it.req_id, offset=it.block_offset, category=it.category(),
                                protected=bool(self._prot_state.get(it.req_id) and self._prot_state[it.req_id].is_protected))
                           for _, it in candidates])
        item = self._transfer_queue.pop(selected[0])
        if state := self._prot_state.get(item.req_id):
            state.record_bypass_resume()
        return item

    def _dispatch_from_priority_queue(self):
        while self._has_write_capacity():
            item = self._select_next_item()
            if item is None:
                break
            self._dispatch_item(item)

    def _dispatch_item(self, item):
        reg = dict(self._stream_registrations[get_base_request_id(item.req_id)])
        count = max(map(len, item.block_chunk), default=0)
        reg['local_block_ids'] = slice_block_ids(
            reg['local_block_ids'], item.block_offset, tuple(map(len, item.block_chunk)))
        tid = item.req_id if item.is_final else f"{item.req_id}#chunk-{item.block_offset}"
        if not item.is_final:
            reg['request_id'] = f"{reg['request_id']}#chunk-{item.block_offset}"
        self._busy_requests.add(item.req_id)
        with self._sending_transfers_lock:
            self._chunk_transfers[tid] = item
        # _eligible already completed the handshake; this call cannot enter
        # the parent's deferred WRITE bypass path.
        item.submitted_at = time.perf_counter()
        if item.block_offset == 0 or item.is_final:
            emit("write_submit", rid=item.req_id, offset=item.block_offset, final=item.is_final)
        self._do_start_push_kv(tid, item.block_chunk, reg)
        with self._sending_transfers_lock:
            handles = self._sending_transfers.get(tid, [])
            active = sum(map(len, self._sending_transfers.values()))
            if not handles:
                self._failed_streams.add(item.req_id)
                logger.error("PUSH_CHUNK_SUBMISSION_FAILED req=%s offset=%d", item.req_id, item.block_offset)
        logger.info("PUSH_CHUNK_WRITE req=%s offset=%d blocks=%d total=%d final=%d active=%d event_ready=1 submit_ms=%.3f queue_ms=%.3f",
                    item.req_id, item.block_offset, count, item.total_blocks, int(item.is_final), active,
                    (time.perf_counter() - item.submitted_at) * 1000,
                    (item.submitted_at - item.inserted_at) * 1000)
