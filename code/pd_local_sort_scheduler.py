"""FCFS consumer of atomic P HTTP ingress windows.

Window timing/sorting now lives in pd_ingress_window, before EngineCore's
input queue. EngineCore registers each complete window in one main-loop utility
call. This scheduler preserves that order and uses vLLM's normal token/KV
budgets and chunked prefill. It never starts a second window or releases a
singleton early. The class names remain stable for existing launch commands.
"""
import os

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler


class LocalSortMixin:
    def __init__(self, *args, **kwargs):
        config = kwargs.get("vllm_config", args[0] if args else None)
        policy = getattr(getattr(config, "scheduler_config", None), "policy", "fcfs")
        if str(policy).lower() != "fcfs":
            raise ValueError("P ingress window order requires FCFS scheduling")
        if os.environ.get("PD_INGRESS_WINDOW") != "1":
            raise ValueError("LocalSortScheduler requires PD_INGRESS_WINDOW=1 and the launcher bootstrap")
        super().__init__(*args, **kwargs)


class LocalSortAsyncScheduler(LocalSortMixin, AsyncScheduler):
    """Production P scheduler; consumes already sorted windows in FCFS order."""


class LocalSortScheduler(LocalSortMixin, Scheduler):
    """Synchronous equivalent."""
