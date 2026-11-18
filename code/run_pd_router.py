#!/usr/bin/env python3
"""P/D disaggregated prefill router (clean version).

Routes each request: P (prefill) → D (decode).
Uses /v1/completions with token-ID list for prompt.
Shares X-Request-Id so vLLM request IDs match for PUSH_REG matching.

Usage:
  python run_pd_router.py [--p-port 8100] [--d-port 8200] [--port 8000]
"""
from __future__ import annotations

import asyncio
import argparse
import json
import logging
import time
import uuid

import aiohttp
import orjson
from aiohttp import web
from pd_pipeline_trace import emit, aiohttp_trace

logger = logging.getLogger("pd_router")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

P_URL: str = "http://127.0.0.1:8100"
D_URL: str = "http://127.0.0.1:8200"
BACKEND_SESSION = web.AppKey("pd_backend_session", aiohttp.ClientSession)


async def backend_connections(app: web.Application):
    # One pool per router, retaining keep-alive connections across requests.
    # No connector queue may hold D behind P: D must register its KV buffers
    # while P's HTTP response is still pending. Keep the previous unbounded
    # aggregate backend concurrency, and do not share backend cookies.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300),
        connector=aiohttp.TCPConnector(limit=0),
        cookie_jar=aiohttp.DummyCookieJar(),
        trace_configs=aiohttp_trace(),
    ) as session:
        app[BACKEND_SESSION] = session
        yield


def prepare_backend_payloads(body: dict, shared_id: str) -> tuple[bytes, bytes]:
    """Encode token IDs once; reuse the encoded array in both backend bodies.

    Both payloads are ready before launching network tasks. aiohttp receives
    bytes via data= and does no further JSON encoding on the sending path.
    Separate metadata still preserves the P/D models and transfer parameters.
    """
    ids = body.get("prompt_token_ids", [])
    prompt = orjson.dumps(ids)
    seed = body.get("seed", 0)
    temperature = body.get("temperature", 0.0)
    stream = bool(body.get("stream", False))
    p_meta = {
        "model": "llama3-p", "max_tokens": 1,
        "temperature": temperature, "seed": seed,
        "kv_transfer_params": {
            "do_remote_decode": True, "remote_engine_id": "",
            "remote_host": "localhost", "remote_port": 5601,
            "tp_size": 1, "pp_size": 1, "dcp_size": 1,
        },
    }
    d_meta = {
        "model": "llama3-d", "max_tokens": body.get("max_tokens", 64),
        "temperature": temperature, "seed": seed, "stream": stream,
        "return_token_ids": bool(body.get("return_token_ids", False)),
        "stream_options": {"include_usage": True} if stream else None,
        "kv_transfer_params": {
            "do_remote_prefill": True, "do_remote_decode": False,
            "remote_block_ids": [], "remote_engine_id": "nixl-p",
            "remote_request_id": shared_id, "remote_host": "localhost",
            "remote_port": 5600, "tp_size": 1, "pp_size": 1, "dcp_size": 1,
            "remote_num_tokens": len(ids), "transfer_mode": "push",
            "remote_prefill_cached_tokens": 0,
        },
    }
    # Each fragment comes from a JSON encoder; no raw user text is spliced
    # into syntax. Sharing these bytes avoids a second traversal of ids.
    prefix = b'{"prompt":' + prompt + b','
    return prefix + orjson.dumps(p_meta)[1:], prefix + orjson.dumps(d_meta)[1:]


async def _post_payload(
    session: aiohttp.ClientSession,
    url: str,
    body: bytes,
    headers: dict | None = None,
    trace_context: dict | None = None,
) -> dict:
    merged = {"Content-Type": "application/json"}
    if headers:
        merged.update(headers)
    emit("router_post", **(trace_context or {}))
    async with session.post(url, data=body, headers=merged,
                            trace_request_ctx=trace_context) as resp:
        emit("router_response", **(trace_context or {}))
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:300]}")
        return await resp.json(loads=orjson.loads)


async def _run_pair(p_task, d_coroutine):
    """Fail promptly on either backend; always reap both tasks."""
    d_task = asyncio.create_task(d_coroutine)
    try:
        done, _ = await asyncio.wait((p_task, d_task), return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            task.result()
        return await p_task, await d_task
    finally:
        for task in (p_task, d_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(p_task, d_task, return_exceptions=True)


async def handle_completion(request: web.Request) -> web.StreamResponse:
    """Route a request through P (prefill) then D (decode)."""
    entered_ns = time.monotonic_ns()
    raw = await request.read()  # Asynchronous I/O; keep HTTP size/error handling.
    read_ns = time.monotonic_ns()
    decode_cpu = time.thread_time_ns()
    try:
        body = orjson.loads(raw)
    except orjson.JSONDecodeError:
        raise web.HTTPBadRequest(text="Invalid JSON body")
    if not isinstance(body, dict) or not isinstance(body.get("prompt_token_ids", []), list):
        raise web.HTTPBadRequest(text="Expected a JSON object with a token-ID list")
    decoded_ns, decoded_cpu = time.monotonic_ns(), time.thread_time_ns()

    router_recv_time = time.perf_counter()
    request_id = body.get("request_id") or str(uuid.uuid4())
    prompt_token_ids = body.get("prompt_token_ids", [])
    want_stream = bool(body.get("stream", False))

    # Shared request ID: both P and D get the same X-Request-Id header.
    # vLLM builds request_id as "{prefix}-{X-Request-Id}" on both sides,
    # so get_base_request_id strips the random suffix and they match.
    shared_id = uuid.uuid4().hex[:16]
    shared_headers = {"X-Request-Id": shared_id}
    context = dict(replay=request.headers.get("X-PD-Replay", ""), rid=request_id,
                   batch=int(request.headers.get("X-PD-Batch", "-1")), shared=shared_id)
    emit("router_body", entered_ns=entered_ns, tokens=len(prompt_token_ids), **context)
    emit("router_read_body_span", start_ns=entered_ns, end_ns=read_ns,
         bytes=len(raw), **context)
    emit("router_decode_span", start_ns=read_ns, end_ns=decoded_ns,
         cpu_ns=decoded_cpu-decode_cpu, **context)

    prepare_ns, cpu_ns = time.monotonic_ns(), time.thread_time_ns()
    try:
        p_body, d_body = prepare_backend_payloads(body, shared_id)
    except (TypeError, ValueError):
        raise web.HTTPBadRequest(text="Invalid backend request fields")
    emit("router_prepare_span", start_ns=prepare_ns, end_ns=time.monotonic_ns(),
         cpu_ns=time.thread_time_ns()-cpu_ns, **context)
    # Release the parsed token array before waiting for inference/SSE results.
    del raw, body, prompt_token_ids

    session = request.app[BACKEND_SESSION]
    t_p = time.perf_counter()

    async def submit_prefill():
        data = await _post_payload(
            session, f"{P_URL}/v1/completions", p_body, headers=shared_headers,
            trace_context=dict(stage="P", **context)
        )
        emit("p_http_done", **context)
        return data, time.perf_counter()

    p_task = asyncio.create_task(submit_prefill())

    t_d = time.perf_counter()
    if want_stream:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream",
                     "Cache-Control": "no-cache"},
        )
        async def forward_decode():
            async with session.post(f"{D_URL}/v1/completions", data=d_body,
                                    headers={"Content-Type": "application/json", **shared_headers},
                                    trace_request_ctx=dict(stage="D", **context)) as upstream:
                if upstream.status != 200:
                    raise RuntimeError(f"D HTTP {upstream.status}: {(await upstream.text())[:300]}")
                await resp.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await resp.write(chunk)

        try:
            (p_data, p_done_at), _ = await _run_pair(p_task, forward_decode())
        except Exception as error:
            logger.error("P/D stream failed req=%s: %s", request_id, error)
            if not resp.prepared:
                raise web.HTTPBadGateway(text=f"P/D request failed: {error}")
            # Once SSE headers were sent, report errors in-band. The load
            # collector rejects this event even if D already emitted DONE.
            await resp.write(('data: ' + json.dumps({'error': str(error)}) + '\n\n').encode())
            await resp.write(b'data: [DONE]\n\n')
        else:
            logger.info("req=%s p=%.1fms e2e=%.1fms (stream)", request_id,
                        (p_done_at - t_p) * 1000,
                        (time.perf_counter() - router_recv_time) * 1000)
        await resp.write_eof()
        return resp
    try:
        (p_data, p_done_at), d_data = await _run_pair(p_task, _post_payload(
            session, f"{D_URL}/v1/completions", d_body, headers=shared_headers
        ))
    except Exception as e:
        logger.error("D decode failed req=%s: %s", request_id, e)
        raise web.HTTPBadGateway(text=f"P/D request failed: {e}")

    p_ms = (p_done_at - t_p) * 1000
    if not p_data.get("kv_transfer_params", {}).get("remote_request_id"):
        logger.warning("P response omitted transfer metadata req=%s", request_id)

    e2e_ms = (time.perf_counter() - router_recv_time) * 1000
    d_ms = (time.perf_counter() - t_d) * 1000
    logger.info("req=%s p=%.1fms d=%.1fms e2e=%.1fms",
                request_id, p_ms, d_ms, e2e_ms)

    d_data.setdefault("pd_metrics", {})
    d_data["pd_metrics"]["ttft_ms"] = None  # Non-stream response cannot measure TTFT.
    d_data["pd_metrics"]["e2e_ms"] = round(e2e_ms, 3)
    d_data["pd_metrics"]["p_ms"] = round(p_ms, 3)
    d_data["pd_metrics"]["d_ms"] = round(d_ms, 3)
    d_data["pd_metrics"]["request_id"] = request_id

    return web.json_response(d_data)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "p_url": P_URL, "d_url": D_URL})


def build_app() -> web.Application:
    app = web.Application()
    app.cleanup_ctx.append(backend_connections)
    app.router.add_post("/v1/chat/completions", handle_completion)
    app.router.add_post("/v1/completions", handle_completion)
    app.router.add_get("/health", handle_health)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="P/D router")
    parser.add_argument("--p-host", default="127.0.0.1")
    parser.add_argument("--p-port", type=int, default=8100)
    parser.add_argument("--d-host", default="127.0.0.1")
    parser.add_argument("--d-port", type=int, default=8200)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global P_URL, D_URL
    P_URL = f"http://{args.p_host}:{args.p_port}"
    D_URL = f"http://{args.d_host}:{args.d_port}"

    logger.info("Router: port=%d  P=%s  D=%s", args.port, P_URL, D_URL)
    app = build_app()
    web.run_app(app, host="0.0.0.0", port=args.port, access_log=None, handler_cancellation=True)


if __name__ == "__main__":
    main()
