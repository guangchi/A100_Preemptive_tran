"""Child-process import hooks for production ingress windows and optional trace.

Enabled by the pair launcher; never changes the installed vLLM package.
"""
import os

_trace = bool(os.environ.get("PD_PIPELINE_TRACE"))
_ingress = os.environ.get("PD_INGRESS_WINDOW") == "1"
if _trace or _ingress:
    import importlib.abc
    import importlib.machinery
    import sys

    _patches = {}

    def _register(module, patch):
        _patches.setdefault(module, []).append(patch)

    if _trace:
        import pd_pipeline_trace as trace
        _register("vllm.v1.engine.core", trace.patch_core)
        _register("vllm.v1.core.sched.scheduler", trace.patch_scheduler)
        _register("vllm.entrypoints.launchers.app", trace.patch_app)
        _register("vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker", trace.patch_native_worker)
        if os.environ.get("PD_API_TIMING") == "1" and os.environ.get("PD_TRACE_ROLE") == "P":
            import pd_api_timing as timing
            _register("uvicorn.protocols.http.httptools_impl", timing.patch_protocol)
            _register("uvicorn.protocols.http.h11_impl", timing.patch_protocol)
            _register("vllm.entrypoints.openai.completion.serving", timing.patch_serving)
    if _ingress:
        import pd_ingress_window as ingress
        _register("vllm.v1.engine.core", ingress.patch_core)
        _register("vllm.v1.engine.core_client", ingress.patch_client)
        _register("vllm.v1.engine.async_llm", ingress.patch_async_llm)
        _register("vllm.entrypoints.openai.completion.serving", ingress.patch_serving)
        _register("vllm.entrypoints.launchers.app", ingress.patch_app)

    class PDFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname not in _patches:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None or spec.loader is None:
                return None
            loader = spec.loader
            patches = _patches.pop(fullname)

            class PDLoader(importlib.abc.Loader):
                def create_module(self, spec):
                    return loader.create_module(spec)

                def exec_module(self, module):
                    loader.exec_module(module)
                    for patch in patches:
                        patch(module)

            spec.loader = PDLoader()
            return spec

    sys.meta_path.insert(0, PDFinder())
