"""Fixed P HTTP arrival windows and atomic EngineCore admission (vLLM 0.33).

No vLLM imports at module load: sitecustomize also runs in spawned workers.
The API event loop owns window membership; EngineCore's main loop alone owns
Request/KV state. A window is one utility message, never a series of ADDs.
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import pairwise
import logging
import math
import os
import time
import uuid

from pd_pipeline_trace import emit

logger = logging.getLogger("vllm.pd_ingress_window")
_ticket = ContextVar("pd_ingress_ticket", default=None)
_detail_timing = os.environ.get("PD_API_TIMING") == "1" and bool(os.environ.get("PD_PIPELINE_TRACE"))


@dataclass(eq=False)
class Ticket:
    window: Window
    arrival_ns: int
    ordinal: int
    future: asyncio.Future
    gate: object = None
    client: object = None
    request: object = None
    skipped: bool = False
    sent: bool = False
    cancelled: bool = False


@dataclass
class Window:
    sequence: int
    opened_ns: int
    deadline_ns: int
    tickets: list[Ticket] = field(default_factory=list)
    timer: object = None
    collection_done: asyncio.Event = field(default_factory=asyncio.Event)


class IngressGate:
    """Fixed [first arrival, first arrival + duration) windows, FIFO dispatch.

    Preparing a request may finish after its window deadline. It still belongs
    to its original window; later windows wait, rather than merging/overtaking.
    """
    def __init__(self, window_ms=5.0, dispatch=None, clock=time.monotonic_ns):
        if not math.isfinite(window_ms) or window_ms <= 0:
            raise ValueError("PD_LOCAL_SORT_WINDOW_MS must be finite and positive")
        self.duration_ns = int(window_ms * 1_000_000)
        if self.duration_ns < 1:
            raise ValueError("window must be at least one nanosecond")
        self.clock = clock
        self.dispatch = dispatch or dispatch_window
        self.frontend = uuid.uuid4().hex
        self.windows = deque()
        self.sequence = 0
        self.event = asyncio.Event()
        self.task = None
        self.closed = False
        self.active = []

    def enter(self):
        if self.closed:
            raise RuntimeError("P ingress window gate is closed")
        loop = asyncio.get_running_loop()
        now = self.clock()
        # Test the actual arrival timestamp, not whether a timer callback ran.
        if not self.windows or now >= self.windows[-1].deadline_ns:
            self.sequence += 1
            window = Window(self.sequence, now, now + self.duration_ns)
            window.timer = loop.call_later(self.duration_ns / 1e9, self._deadline, window)
            self.windows.append(window)
        window = self.windows[-1]
        ticket = Ticket(window, now, len(window.tickets), loop.create_future())
        ticket.gate = self
        window.tickets.append(ticket)
        emit("ingress_arrive", window=window.sequence, ordinal=ticket.ordinal,
             arrival_ns=now, deadline_ns=window.deadline_ns)
        if self.task is None:
            self.task = asyncio.create_task(self._run(), name="pd-ingress-windows")
        self.event.set()
        return ticket

    def _deadline(self, window):
        remaining = window.deadline_ns - self.clock()
        if remaining > 0:
            # Event loops may run timers slightly early. Never lose the wakeup
            # if the strict monotonic deadline has not actually arrived yet.
            window.timer = asyncio.get_running_loop().call_later(
                remaining / 1e9, self._deadline, window)
        else:
            window.collection_done.set()
            self.event.set()

    async def wait_collection(self, ticket):
        """Keep preprocessing out of this window's collection interval.

        Waiters yield immediately, so another ready HTTP task can register its
        arrival before JSON/model validation of the first member begins.
        This does not isolate the loop from other windows or HTTP callbacks.
        """
        window = ticket.window
        if self.clock() >= window.deadline_ns:
            window.collection_done.set()
        await window.collection_done.wait()
        if self.closed:
            raise RuntimeError("P ingress window gate stopped")

    async def submit(self, ticket, client, request):
        if ticket.request is not None or ticket.skipped:
            raise ValueError("one P HTTP request must generate exactly one engine request")
        ticket.client, ticket.request = client, request
        emit("window_admit", rid=request.request_id, window=ticket.window.sequence,
             tokens=len(request.prompt_token_ids), admitted_ns=ticket.arrival_ns,
             deadline_ns=ticket.window.deadline_ns, source="p_http_ingress")
        self.event.set()
        try:
            result = await asyncio.shield(ticket.future)
        except asyncio.CancelledError:
            ticket.cancelled = True
            self.skip(ticket)
            if ticket.sent and ticket.future.done():
                # Cancellation can race with the dispatcher completing the
                # future, after its own cancelled-member check has finished.
                await client.abort_requests_async([request.request_id])
            raise
        if isinstance(result, BaseException):
            raise result

    def skip(self, ticket):
        # Before dispatch, invalid/disconnected HTTP requests must not hold up
        # their neighbours. After dispatch, abort only after the ADD is acked.
        if not ticket.sent and not ticket.future.done():
            ticket.skipped = True
            ticket.future.set_result(None)
        self.event.set()

    async def _run(self):
        try:
            while True:
                await self.event.wait()
                self.event.clear()
                while self.windows:
                    window = self.windows[0]
                    if self.clock() < window.deadline_ns:
                        break
                    if any(t.request is None and not t.skipped for t in window.tickets):
                        break
                    self.windows.popleft()
                    window.timer.cancel()
                    tickets = [t for t in window.tickets if not t.skipped]
                    # Closed windows cannot acquire new members. Break the
                    # window -> ticket -> window cycle, releasing prompt data
                    # promptly once the HTTP handlers finish.
                    window.tickets.clear()
                    tickets.sort(key=lambda t: (len(t.request.prompt_token_ids), t.ordinal))
                    if not tickets:
                        continue
                    self.active = tickets
                    error = None
                    try:
                        if any(t.client is not tickets[0].client for t in tickets):
                            raise ValueError("one ingress gate requires a single EngineCore client")
                        for t in tickets:
                            t.sent = True
                        await self.dispatch(self.frontend, window, tickets)
                        cancelled = [t.request.request_id for t in tickets if t.cancelled]
                        if cancelled:
                            await tickets[0].client.abort_requests_async(cancelled)
                    except Exception as exc:
                        error = exc
                        logger.exception("P ingress window %d failed", window.sequence)
                    for t in tickets:
                        if not t.future.done():
                            t.future.set_result(error)
                    self.active = []
        finally:
            self._stop_waiters()

    def _stop_waiters(self):
        self.closed = True
        error = RuntimeError("P ingress window gate stopped")
        for window in self.windows:
            window.timer.cancel()
            window.collection_done.set()
        for t in self.active + [t for w in self.windows for t in w.tickets]:
            if not t.future.done():
                t.future.set_result(error)

    async def close(self):
        self.closed = True
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        # A task cancelled before its first turn never executes its finally.
        self._stop_waiters()


def encode_requests(requests):
    from vllm.v1.serial_utils import MsgpackEncoder
    return [bytes(frame) for frame in MsgpackEncoder().encode(requests)]


def decode_requests(frames):
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder
    return MsgpackDecoder(list[EngineCoreRequest]).decode(frames)


async def dispatch_window(frontend, window, tickets):
    metadata = dict(frontend=frontend, window=window.sequence,
                    opened_ns=window.opened_ns, deadline_ns=window.deadline_ns,
                    members=[dict(rid=t.request.request_id, arrival_ns=t.arrival_ns)
                             for t in tickets])
    # Existing NIXL/sampling fields use vLLM's own typed serialization. The
    # outer utility contains opaque frames, so generic msgpack loses no types.
    frames = encode_requests([t.request for t in tickets])
    emit("ingress_window_submit", **metadata)
    await tickets[0].client.call_utility_async("pd_add_ingress_window", metadata, frames)


def add_ingress_window(core, metadata, frames):
    """Runs synchronously as ONE utility in EngineCoreProc's main input loop.

    All members register before the next schedule() call. The normal scheduler
    then consumes FCFS order subject to token/KV budgets and chunked prefill.
    """
    requests = decode_requests(frames)
    members = metadata["members"]
    if [r.request_id for r in requests] != [m["rid"] for m in members]:
        raise ValueError("ingress window request IDs do not match metadata")
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("duplicate ingress request ID")
    keys = [(len(r.prompt_token_ids), m["arrival_ns"]) for r, m in zip(requests, members)]
    # The API already sorted this window. Validate its order in one pass,
    # rather than sorting a second copy in EngineCore's compute/input loop.
    if any(left > right for left, right in pairwise(keys)):
        raise ValueError("ingress window is not sorted by prompt length / arrival")
    if any(not metadata["opened_ns"] <= m["arrival_ns"] < metadata["deadline_ns"]
           for m in members):
        raise ValueError("request falls outside its fixed ingress window")
    # No scheduling or timer thread can run between these registrations.
    prepared = [core.preprocess_add_request(r) for r in requests]
    attempted = []
    try:
        for rank, ((request, wave), member) in enumerate(zip(prepared, members)):
            attempted.append(request.request_id)
            if core._reject_add_in_shutdown(request) or core._reject_add_while_paused(request):
                continue
            core.add_request(request, wave)
            released = time.monotonic_ns()
            emit("window_release", rid=request.request_id, window=metadata["window"],
                 rank=rank, admitted_ns=member["arrival_ns"],
                 deadline_ns=metadata["deadline_ns"], released_ns=released,
                 source="p_http_ingress")
            logger.info("PD ingress admission: window=%d rank=%d req=%s tokens=%d "
                        "admitted_ns=%d deadline_ns=%d released_ns=%d",
                        metadata["window"], rank, request.request_id,
                        request.num_prompt_tokens, member["arrival_ns"],
                        metadata["deadline_ns"], released)
    except Exception:
        core.abort_requests(attempted)
        raise
    emit("ingress_window_registered", window=metadata["window"], count=len(prepared))
    return len(prepared)


def patch_core(module):
    module.EngineCoreProc.pd_add_ingress_window = add_ingress_window


def patch_client(module):
    cls = module.AsyncMPClient
    original = cls.add_request_async

    async def add(self, request):
        if request.abort_immediately:
            return await original(self, request)  # vLLM remote KV cleanup marker
        ticket = _ticket.get()
        if ticket is None:
            raise ValueError("P window sorting requires /v1/completions HTTP ingress")
        if type(self) is not cls or len(self.core_engines) != 1:
            raise ValueError("P ingress windows require one API process and DP=1")
        if (request.mm_features or request.prompt_embeds is not None or request.resumable
                or request.prompt_token_ids is None or request.sampling_params is None
                or request.sampling_params.n != 1
                or request.sampling_params.structured_outputs is not None):
            raise ValueError("P ingress windows support one text/token prompt, n=1, without structured output")
        request.client_index = self.client_index
        return await ticket.gate.submit(ticket, self, request)

    cls.add_request_async = add


def patch_async_llm(module):
    original = module.AsyncLLM._add_request

    async def add(self, request, *args, **kwargs):
        try:
            return await original(self, request, *args, **kwargs)
        except BaseException:
            # vLLM registers its output collector before awaiting engine input.
            # A failed/cancelled ingress wait must also release that collector.
            await self.abort(request.request_id, internal=True)
            raise

    module.AsyncLLM._add_request = add


def patch_serving(module):
    """Check fanout AFTER FastAPI validation, BEFORE rendering/generation.

    CompletionRequest has already validated its homogeneous prompt union, so
    inspecting one element distinguishes token IDs from multi-prompt lists.
    No second JSON decode or Python traversal of every token is required.
    """
    cls = module.OpenAIServingCompletion
    original = cls.render_completion_request

    async def render(self, request, *args, **kwargs):
        if _ticket.get() is not None:
            prompt = request.prompt
            single = (isinstance(prompt, str) or
                      (isinstance(prompt, list) and bool(prompt) and type(prompt[0]) is int))
            if (not single or request.n != 1
                    or getattr(request, "best_of", None) not in (None, 1)
                    or request.use_beam_search):
                return self.create_error_response(
                    "P ingress window sorting requires a single text/token prompt and n=1")
        return await original(self, request, *args, **kwargs)

    cls.render_completion_request = render


class IngressASGI:
    def __init__(self, app, window_ms=5.0):
        self.app = app
        self.gate = IngressGate(window_ms)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            stop_probe = None
            try:
                if _detail_timing:
                    from pd_api_timing import start_loop_probe
                    stop_probe = start_loop_probe()
                return await self.app(scope, receive, send)
            finally:
                if stop_probe:
                    stop_probe()
                await self.gate.close()
        if (scope["type"] != "http" or scope.get("method") != "POST"
                or scope.get("path") != "/v1/completions"):
            return await self.app(scope, receive, send)
        ticket = self.gate.enter()  # Before reading body or tokenizing.
        shared = dict(scope.get("headers", [])).get(b"x-request-id", b"").decode("latin-1")
        emit("ingress_http", shared=shared, window=ticket.window.sequence,
             arrival_ns=ticket.arrival_ns, deadline_ns=ticket.window.deadline_ns)
        token = _ticket.set(ticket)
        try:
            fields = dict(shared=shared, window=ticket.window.sequence, ordinal=ticket.ordinal)
            await self.gate.wait_collection(ticket)
            emit("ingress_preprocess_start", **fields)
            # Let Starlette/FastAPI read and decode the original body exactly
            # once. In particular, do not buffer and parse it in this wrapper.
            if _detail_timing:
                emit("api_ingress_forward", **fields)
                original_receive = receive
                body_start, body_bytes = None, 0

                async def observed_receive():
                    nonlocal body_start, body_bytes
                    if body_start is None:
                        body_start = time.monotonic_ns()
                    message = await original_receive()
                    if message["type"] == "http.request":
                        body_bytes += len(message.get("body", b""))
                        if not message.get("more_body", False):
                            emit("api_read_body_span", start_ns=body_start,
                                 end_ns=time.monotonic_ns(), bytes=body_bytes, **fields)
                    return message

                receive = observed_receive
            return await self.app(scope, receive, send)
        finally:
            self.gate.skip(ticket)
            _ticket.reset(token)


def patch_app(module):
    build = module.build_app

    def build_with_ingress(*args, **kwargs):
        config = args[0] if args else kwargs["args"]
        for name in ("api_server_count", "data_parallel_size", "data_parallel_size_local"):
            if getattr(config, name, None) not in (None, 1):
                raise ValueError("P ingress windows require one API process and DP=1")
        app = build(*args, **kwargs)
        app.middleware_stack = IngressASGI(
            app.middleware_stack or app.build_middleware_stack(),
            float(os.environ.get("PD_LOCAL_SORT_WINDOW_MS", "10")))
        logger.info("PD P HTTP ingress windows enabled: fixed %sms, atomic EngineCore admission",
                    os.environ.get("PD_LOCAL_SORT_WINDOW_MS", "10"))
        return app

    module.build_app = build_with_ingress
