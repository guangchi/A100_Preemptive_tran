#!/usr/bin/env python3
"""Sort-only ingress timing: three fresh starts, 50/25/50/25 per start."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import run_pd_local_sort_comparison as runner
from pd_experiment_paths import DEFAULT_WORKLOAD, reference_path


def main(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "runner_pid.txt").write_text(str(os.getpid()))
    workload = root / "workload.jsonl"
    reference = root / "reference_outputs_a100.csv"
    shutil.copy2(DEFAULT_WORKLOAD, workload)
    shutil.copy2(reference_path(), reference)
    shutil.copytree(runner.CODE_DIR, root / "source", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(p.relative_to(root / "source")): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root / "source").rglob("*") if p.is_file()}
    (root / "source_sha256.json").write_text(json.dumps(hashes, indent=2))
    manifest = dict(gpu_prefill=runner.P_GPU, gpu_decode=runner.D_GPU,
                    scheme="sort_only", fresh_starts=3, formal_order=[50,25,50,25],
                    warmup_order=[50,25],
                    chunk_size_blocks=int(os.environ.get("CHUNK_SIZE_BLOCKS", "8")),
                    window_ms=float(os.environ.get("PD_LOCAL_SORT_WINDOW_MS", "10")),
                    stream_max_in_flight=int(os.environ.get("PD_STREAM_MAX_IN_FLIGHT", "1")),
                    stream_poll_ms=float(os.environ.get("PD_STREAM_POLL_MS", "0.1")),
                    api_timing=os.environ.get("PD_API_TIMING") == "1",
                    workload_sha256=hashlib.sha256(workload.read_bytes()).hexdigest())
    (root / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    os.environ["PD_PIPELINE_TRACE"] = str(root / "trace")
    runner.check_experiment_inputs(str(workload), False, reference)
    progress = []
    for startup in range(1, 4):
        index = 2*startup-1
        logs = root / f"logs_round{index}_sort_only"
        logs.mkdir()
        # Two formal pairs share one service. A relative symlink lets the
        # existing trace analyzer resolve the second pair's common P log.
        (root / f"logs_round{index+1}_sort_only").symlink_to(logs.name, target_is_directory=True)
        router = None
        try:
            runner.launch_scheme("sort_only", logs)
            router = runner.start_router(logs)
            for gap in (50, 25):
                assert runner.run_load(str(workload), runner.SCENARIOS[f"{gap}ms"],
                                       str(logs / f"warmup_{gap}ms.csv"), reference), "warmup validation failed"
            for pair in (1, 2):
                for gap in (50, 25):
                    run_index = index + pair-1
                    path = root / "runs" / f"sort_only_{gap}ms_run{run_index}" / "requests.csv"
                    started = time.monotonic_ns()
                    valid = runner.run_load(str(workload), runner.SCENARIOS[f"{gap}ms"], str(path), reference)
                    progress.append(dict(startup=startup, pair=pair, round=run_index,
                                         scenario=f"{gap}ms", scheme="sort_only", csv=str(path),
                                         success=valid, started_ns=started, ended_ns=time.monotonic_ns()))
                    (root / "progress.json").write_text(json.dumps(progress, indent=2))
                    if not valid:
                        raise RuntimeError(f"Formal validation failed: {path}")
        finally:
            if router is not None:
                runner.stop_router(router)
            runner.stop_pair(logs)
    (root / "experiment_summary.json").write_text(json.dumps(dict(complete=True, runs=progress), indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
