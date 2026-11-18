"""Opt-in host pipeline timestamps. No prompt/output payloads are recorded.

PD_PIPELINE_TRACE selects a directory; one append-only JSONL file per PID.
All timestamps use the same host CLOCK_MONOTONIC, across processes/threads.
"""
import json
import os
import time

_directory = os.environ.get("PD_PIPELINE_TRACE")
_pid = None
_fd = None


def emit(event, **fields):
    if not _directory:
        return
    stamp = time.monotonic_ns()
    global _pid, _fd
    pid = os.getpid()
    if pid != _pid:
        if _fd is not None:
            os.close(_fd)
        os.makedirs(_directory, exist_ok=True)
        _fd = os.open(os.path.join(_directory, f"{pid}.jsonl"),
                      os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        _pid = pid
    record = dict(t_ns=stamp, pid=pid, role=os.environ.get("PD_TRACE_ROLE", "router"),
                  event=event, **fields)
    os.write(_fd, (json.dumps(record, separators=(",", ":")) + "\n").encode())


def aiohttp_trace():
    """Record when headers actually leave the client request machinery."""
    import aiohttp
    trace = aiohttp.TraceConfig()

    async def headers_sent(session, ctx, params):
        emit("http_headers_sent", **(ctx.trace_request_ctx or {}))

    trace.on_request_headers_sent.append(headers_sent)
    return [trace] if _directory else []


def patch_input_queue(queue, add_type):
    """Keep queue payload/API unchanged, observing both ends of ADD messages."""
    put, get = queue.put, queue.get

    def request_id(item):
        if item[0] == add_type:
            return item[1][0].request_id
        return None

    def traced_put(item, *args, **kwargs):
        rid = request_id(item)
        if rid is not None:
            emit("engine_enqueue", rid=rid, depth=queue.qsize())
        return put(item, *args, **kwargs)

    def traced_get(*args, **kwargs):
        item = get(*args, **kwargs)
        rid = request_id(item)
        if rid is not None:
            emit("engine_dequeue", rid=rid, depth=queue.qsize())
        return item

    queue.put, queue.get = traced_put, traced_get


def patch_core(module):
    core, proc = module.EngineCore, module.EngineCoreProc
    preprocess, add = core.preprocess_add_request, core.add_request
    sockets, step = proc.process_input_sockets, proc._process_engine_step

    def traced_preprocess(self, request):
        emit("engine_socket", rid=request.request_id)
        return preprocess(self, request)

    def traced_add(self, request, *args, **kwargs):
        emit("engine_add", rid=request.request_id, tokens=request.num_prompt_tokens)
        return add(self, request, *args, **kwargs)

    def traced_sockets(self, *args, **kwargs):
        patch_input_queue(self.input_queue, module.EngineCoreRequestType.ADD)
        emit("trace_core_ready")
        return sockets(self, *args, **kwargs)

    def traced_step(self):
        start = time.monotonic_ns()
        result = step(self)
        end = time.monotonic_ns()
        # Omit idle iterations. Long iterations can delay input queue draining.
        if end - start > 5_000_000:
            emit("engine_step", start_ns=start, end_ns=end, executed=result)
        return result

    core.preprocess_add_request, core.add_request = traced_preprocess, traced_add
    proc.process_input_sockets, proc._process_engine_step = traced_sockets, traced_step


def patch_scheduler(module):
    schedule = module.Scheduler.schedule

    def traced_schedule(self, *args, **kwargs):
        output = schedule(self, *args, **kwargs)
        if output.num_scheduled_tokens:
            emit("scheduled", tokens=output.num_scheduled_tokens)
        return output

    module.Scheduler.schedule = traced_schedule


class PipelineASGI:
    """Observe API arrival/body delivery without buffering or changing messages."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/v1/completions":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        shared = headers.get(b"x-request-id", b"").decode()
        emit("api_enter", shared=shared)

        async def traced_receive():
            message = await receive()
            if message["type"] == "http.request" and not message.get("more_body", False):
                emit("api_body", shared=shared)
            return message

        return await self.app(scope, traced_receive, send)


def patch_app(module):
    build = module.build_app

    def traced_build(*args, **kwargs):
        app = build(*args, **kwargs)
        # Some vLLM middleware builds the stack during build_app already.
        # Keep the FastAPI object/attributes, wrapping its ASGI stack directly.
        app.middleware_stack = PipelineASGI(app.middleware_stack or app.build_middleware_stack())
        return app

    module.build_app = traced_build


def patch_native_worker(module):
    """Observe original NIXL whole-request path for baseline stage comparison."""
    worker = module.NixlPushConnectorWorker
    start, write, poll = worker.start_load_kv, worker._do_start_push_kv, worker._pop_done_transfers

    def traced_start(self, metadata, *args, **kwargs):
        for rid in metadata.push_finished_blocks:
            emit("native_blocks_published", rid=rid)
        return start(self, metadata, *args, **kwargs)

    def traced_write(self, request_id, *args, **kwargs):
        # Streaming partial transfers also call the base method. Record only
        # IDs without a chunk suffix; streaming records remain separately named.
        if "#chunk-" not in request_id:
            emit("native_write_call", rid=request_id)
        return write(self, request_id, *args, **kwargs)

    def traced_poll(self, transfers):
        result = poll(self, transfers)
        if transfers is self._sending_transfers:
            for rid in result[0]:
                if "#chunk-" not in rid:
                    emit("native_done_observed", rid=rid)
        return result

    worker.start_load_kv, worker._do_start_push_kv, worker._pop_done_transfers = traced_start, traced_write, traced_poll
