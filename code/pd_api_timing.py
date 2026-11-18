"""Opt-in API timing, enabled by PD_API_TIMING=1 plus pipeline trace.

Protocol timestamps are userspace HTTP processing times, not kernel arrivals.
Synchronous spans record thread CPU; asynchronous spans record wall time only.
"""
import asyncio
from contextlib import contextmanager
import time

from pd_pipeline_trace import emit


@contextmanager
def sync_span(phase, **fields):
    start, cpu = time.monotonic_ns(), time.thread_time_ns()
    try:
        yield
    finally:
        end, cpu_end = time.monotonic_ns(), time.thread_time_ns()
        emit("api_sync_span", phase=phase, start_ns=start, end_ns=end,
             cpu_ns=cpu_end-cpu, **fields)


def shared_id(scope):
    return dict(scope.get("headers", [])).get(b"x-request-id", b"").decode("latin-1")


def start_loop_probe():
    loop = asyncio.get_running_loop()
    previous = [time.monotonic_ns(), time.thread_time_ns()]
    interval = 1_000_000

    def tick():
        now, cpu = time.monotonic_ns(), time.thread_time_ns()
        elapsed = now-previous[0]
        if elapsed >= 2_000_000:
            emit("api_loop_gap", start_ns=previous[0], end_ns=now,
                 elapsed_ns=elapsed, cpu_ns=cpu-previous[1],
                 late_ns=elapsed-interval)
        previous[:] = [time.monotonic_ns(), time.thread_time_ns()]
        handle[0] = loop.call_later(.001, tick)

    handle = [loop.call_later(.001, tick)]
    return lambda: handle[0].cancel()


def patch_protocol(module):
    cycle = module.RequestResponseCycle
    init, run = cycle.__init__, cycle.run_asgi

    def timed_init(self, scope, *args, **kwargs):
        init(self, scope, *args, **kwargs)
        if scope.get("path") == "/v1/completions":
            emit("http_protocol_headers", shared=shared_id(scope), protocol=module.__name__)

    async def timed_run(self, *args, **kwargs):
        if self.scope.get("path") == "/v1/completions":
            emit("http_asgi_task_start", shared=shared_id(self.scope))
        return await run(self, *args, **kwargs)

    cycle.__init__, cycle.run_asgi = timed_init, timed_run


def patch_serving(module):
    cls = module.OpenAIServingCompletion
    original = cls.render_completion_request

    async def timed_render(self, request, *args, **kwargs):
        # Request ID is assigned after rendering; retain correlation with the
        # ingress ContextVar which propagates through API child tasks.
        from pd_ingress_window import _ticket
        ticket = _ticket.get()
        fields = dict(window=ticket.window.sequence, ordinal=ticket.ordinal) if ticket else {}
        start = time.monotonic_ns()
        try:
            return await original(self, request, *args, **kwargs)
        finally:
            emit("api_render_span", start_ns=start, end_ns=time.monotonic_ns(), **fields)

    cls.render_completion_request = timed_render
