"""Validate that diagnostic wrappers preserve queue and ASGI semantics."""
import asyncio
from queue import Queue, Empty
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pd_pipeline_trace import PipelineASGI, patch_app, patch_input_queue, patch_native_worker


class PipelineTraceTests(unittest.TestCase):
    def test_native_worker_hooks_preserve_transfer_results_and_call_arguments(self):
        class Worker:
            def __init__(self):
                self._sending_transfers = {"r": [123]}

            def start_load_kv(self, metadata):
                return metadata

            def _do_start_push_kv(self, request_id, blocks, reg):
                return request_id, blocks, reg

            def _pop_done_transfers(self, transfers):
                transfers.clear()
                return {"r"}, set()

        module = SimpleNamespace(NixlPushConnectorWorker=Worker)
        patch_native_worker(module)
        obj = Worker()
        meta = SimpleNamespace(push_finished_blocks={"r": [1]})
        with patch("pd_pipeline_trace.emit") as emit:
            self.assertIs(obj.start_load_kv(meta), meta)
            self.assertEqual(obj._do_start_push_kv("r", [1], {"d": 2}), ("r", [1], {"d": 2}))
            self.assertEqual(obj._pop_done_transfers(obj._sending_transfers), ({"r"}, set()))
        self.assertEqual([c.args[0] for c in emit.call_args_list],
                         ["native_blocks_published", "native_write_call", "native_done_observed"])

    def test_app_wrapper_handles_prebuilt_stack_and_keeps_fastapi_attributes(self):
        from fastapi import FastAPI
        app = FastAPI()
        app.state.marker = "kept"
        app.middleware_stack = app.build_middleware_stack()
        module = SimpleNamespace(build_app=lambda: app)
        patch_app(module)
        wrapped = module.build_app()
        self.assertIs(wrapped, app)
        self.assertEqual(wrapped.state.marker, "kept")
        self.assertIsInstance(wrapped.middleware_stack, PipelineASGI)

    def test_queue_nowait_and_blocking_share_observers_without_changing_payload(self):
        queue = Queue()
        item = ("ADD", (SimpleNamespace(request_id="r"), 0))
        with patch("pd_pipeline_trace.emit") as emit:
            patch_input_queue(queue, "ADD")
            queue.put_nowait(item)
            self.assertIs(queue.get_nowait(), item)
            queue.put(("CONTROL", None))
            self.assertEqual(queue.get(), ("CONTROL", None))
            with self.assertRaises(Empty):
                queue.get_nowait()
        self.assertEqual([c.args[0] for c in emit.call_args_list],
                         ["engine_enqueue", "engine_dequeue"])

    def test_asgi_keeps_body_chunks_and_response_messages_intact(self):
        chunks = [dict(type="http.request", body=b"part1", more_body=True),
                  dict(type="http.request", body=b"part2", more_body=False)]
        responses, received = [], []

        async def receive():
            return chunks[len(received)]

        async def send(message):
            responses.append(message)

        async def app(scope, receive, send):
            for _ in chunks:
                received.append(await receive())
            await send(dict(type="http.response.body", body=b"SSE"))

        with patch("pd_pipeline_trace.emit") as emit:
            asyncio.run(PipelineASGI(app)(dict(type="http", path="/v1/completions",
                        headers=[(b"x-request-id", b"shared")]), receive, send))
        self.assertEqual(received, chunks)
        self.assertEqual(responses, [dict(type="http.response.body", body=b"SSE")])
        self.assertEqual([c.args[0] for c in emit.call_args_list], ["api_enter", "api_body"])


if __name__ == "__main__":
    unittest.main()
