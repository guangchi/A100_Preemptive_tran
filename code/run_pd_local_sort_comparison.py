#!/usr/bin/env python3
"""Run independently switched sorting/preemption schemes on a local GPU pair.

Runs baseline / sort_only / preempt_only / new in rotating order across 3 rounds,
with 2 arrival-gap scenarios (25ms and 50ms batch intervals).
Each scheme is started fresh per round; results go into timestamped dirs.

Usage:
  python run_pd_local_sort_comparison.py \\
      [--result-dir results/nvlink-local-sort-comparison-YYYYMMDD] \\
      [--workload workload.sweep-interleaved-8s3m4l.jsonl] \\
      [--rounds 3] [--warmup] [--dry-run]

Set PD_P_GPU and PD_D_GPU to the physical NVIDIA GPU indices.
Each round rotates the selected schemes and alternates the scenario order.

Use --schemes to select a subset or include stream_fifo (both switches off,
same streaming transport), or old/new for historical comparison.

Each scheme run:
  1. Launch P/D via run_pd_pair.sh
  2. Warmup: run workload once per scenario (results discarded)
  3. Formal: run workload 1× per scenario, save to CSV
  4. Stop P/D
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from pd_experiment_paths import CODE_DIR, DEFAULT_WORKLOAD, RESULTS_DIR, reference_path
from validate_pd_run import validate_run

logger = logging.getLogger("pd_comparison")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

PYTHON = sys.executable
RUN_PAIR_SH = str(CODE_DIR / "run_pd_pair.sh")
RUN_LOAD_PY = str(CODE_DIR / "run_pd_load.py")
RUN_ROUTER_PY = str(CODE_DIR / "run_pd_router.py")

P_GPU = int(os.environ.get("PD_P_GPU", "6"))
D_GPU = int(os.environ.get("PD_D_GPU", "7"))

SCHEMES = ["baseline", "sort_only", "preempt_only", "new"]
SCENARIOS = {
    "25ms": [0, 25, 50],
    "50ms": [0, 50, 100],
}

ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "8000"))
P_PORT = int(os.environ.get("P_PORT", "8100"))
D_PORT = int(os.environ.get("D_PORT", "8200"))
ROUTER_URL = f"http://127.0.0.1:{ROUTER_PORT}"


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def read_references(path: Path) -> dict:
    with path.open(newline="") as file:
        return {r["request_id"]: r for r in csv.DictReader(file)}


def check_experiment_inputs(workload_path: str, dry_run: bool, reference: Path | None = None) -> None:
    with open(workload_path) as workload_file:
        requests = [json.loads(line) for line in workload_file if line.strip()]
    if len(requests) != 15 or len({r["request_id"] for r in requests}) != 15:
        raise RuntimeError("The comparison requires 15 distinct requests in three groups of five")
    invalid = [r.get("request_id", "?") for r in requests if r.get("max_tokens") != 64]
    if invalid:
        raise RuntimeError(
            "All experiment requests must set max_tokens=64; invalid requests: "
            + ", ".join(invalid[:10])
        )
    logger.info("Validated %d requests with max_tokens=64", len(requests))
    reference = reference or reference_path()
    references = read_references(reference)
    for request in requests:
        rid = request["request_id"]
        if rid not in references:
            raise RuntimeError(f"Missing reference output for {rid} in {reference}")
        ids = json.loads(references[rid]["output_token_ids"])
        if not isinstance(ids, list) or len(ids) != 64 or not all(type(i) is int for i in ids):
            raise RuntimeError(f"Invalid reference token IDs for {rid} in {reference}")
    if len({ROUTER_PORT, P_PORT, D_PORT}) != 3 or not all(
        1 <= port <= 65535 for port in (ROUTER_PORT, P_PORT, D_PORT)
    ):
        raise RuntimeError("P_PORT, D_PORT and ROUTER_PORT must be distinct valid TCP ports")

    if dry_run:
        return

    try:
        visible = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
        gpu_indices = {int(line.strip()) for line in visible if line.strip().isdigit()}
        if P_GPU == D_GPU or not {P_GPU, D_GPU}.issubset(gpu_indices):
            raise RuntimeError(f"GPU {P_GPU} and {D_GPU} are not both visible; found {sorted(gpu_indices)}")
        processes = subprocess.run(
            ["nvidia-smi", "-i", f"{P_GPU},{D_GPU}", "--query-compute-apps=pid",
             "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"Cannot verify GPU {P_GPU}/{D_GPU} availability: {exc}") from exc
    pids = [line.strip() for line in processes if line.strip()]
    if pids:
        raise RuntimeError(f"GPU {P_GPU}/{D_GPU} are occupied by compute process(es): {', '.join(pids)}")
    logger.info("GPU %s/%s are visible and have no compute processes", P_GPU, D_GPU)
    nvlink_snapshot()


# ── Service lifecycle ─────────────────────────────────────────────────────── #

def launch_scheme(scheme: str, log_dir: Path) -> subprocess.Popen:
    """Launch P/D pair for the given scheme."""
    env = os.environ.copy()
    env["LOG_DIR"] = str(log_dir)
    env["PD_PYTHON"] = PYTHON
    env.update(P_PORT=str(P_PORT), D_PORT=str(D_PORT), ROUTER_PORT=str(ROUTER_PORT))
    env["PD_UCX_TLS"] = "cuda,tcp"
    env["PD_UCX_LOG_LEVEL"] = "info"
    feature_flags = {
        "baseline": ("0", "0"),
        "sort_only": ("0", "1"),
        "preempt_only": ("1", "0"),
        "new": ("1", "1"),
        "stream_fifo": ("0", "0"),
    }
    if scheme in feature_flags:
        env["PD_PRIORITY_PREEMPTION"], env["PD_LOCAL_SORT"] = feature_flags[scheme]

    proc = subprocess.Popen(
        ["bash", RUN_PAIR_SH, scheme],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait for the script to finish (it blocks until both servers are ready)
    try:
        stdout, _ = proc.communicate(timeout=300)
    except BaseException:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise
    (log_dir / "launch.log").write_text(stdout)
    if proc.returncode != 0:
        logger.error("launch_scheme(%s) failed:\n%s", scheme, stdout[-2000:])
        raise RuntimeError(f"Failed to launch scheme={scheme}")
    logger.info("[%s] %s P/D ready", ts(), scheme)
    return proc


def nvlink_data_kib(gpu: int) -> dict[int, dict[str, int]]:
    output = subprocess.run(
        ["nvidia-smi", "nvlink", "-i", str(gpu), "-gt", "d"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout
    counters: dict[int, dict[str, int]] = {}
    for link, direction, value in re.findall(
        r"Link\s+(\d+): Data (Tx|Rx): (\d+) KiB", output
    ):
        counters.setdefault(int(link), {})[direction] = int(value)
    if not counters or any(counts.keys() != {"Tx", "Rx"} for counts in counters.values()):
        raise RuntimeError(f"No usable NVLink Tx/Rx counters for GPU {gpu}")
    return counters


def nvlink_snapshot() -> dict[str, dict[int, dict[str, int]]]:
    return {str(gpu): nvlink_data_kib(gpu) for gpu in (P_GPU, D_GPU)}


def nvlink_delta(before, after) -> dict[str, float]:
    tx_kib = sum(after[str(P_GPU)][link]["Tx"] - counts["Tx"]
                 for link, counts in before[str(P_GPU)].items())
    rx_kib = sum(after[str(D_GPU)][link]["Rx"] - counts["Rx"]
                 for link, counts in before[str(D_GPU)].items())
    return {f"gpu{P_GPU}_tx_gb": tx_kib * 1024 / 1e9,
            f"gpu{D_GPU}_rx_gb": rx_kib * 1024 / 1e9}


def stop_pair(log_dir: Path) -> None:
    env = os.environ.copy()
    env["LOG_DIR"] = str(log_dir)
    subprocess.run(["bash", RUN_PAIR_SH, "stop"], env=env, check=False)
    for _ in range(30):
        busy = subprocess.run(
            ["nvidia-smi", "-i", f"{P_GPU},{D_GPU}", "--query-compute-apps=pid", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if not busy:
            return
        time.sleep(1)
    raise RuntimeError(f"GPU{P_GPU}/{D_GPU} not released after stop: {busy}")


def start_router(log_dir: Path) -> subprocess.Popen:
    """Start the router process."""
    router_log = log_dir / "router.log"
    proc = subprocess.Popen(
        [
            PYTHON, RUN_ROUTER_PY,
            "--p-port", str(P_PORT),
            "--d-port", str(D_PORT),
            "--port", str(ROUTER_PORT),
        ],
        stdout=open(router_log, "w"),
        stderr=subprocess.STDOUT,
    )
    # Wait for router to be ready
    for _ in range(30):
        time.sleep(1)
        try:
            import urllib.request
            urllib.request.urlopen(f"{ROUTER_URL}/health", timeout=2)
            logger.info("[%s] Router ready", ts())
            return proc
        except Exception:
            pass
    raise RuntimeError("Router did not start in 30s")


def stop_router(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── Load runner ──────────────────────────────────────────────────────────── #

def run_load(
    workload_path: str,
    gaps_ms: list[int],
    output_csv: str,
    reference: Path | None = None,
) -> bool:
    """Run the load generator and return True on success."""
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now().isoformat()
    nvlink_before = nvlink_snapshot()
    replay_id = uuid.uuid4().hex
    reference = reference or reference_path()
    load_env = {**os.environ, "PD_TRACE_REPLAY": replay_id, "PD_TRACE_ROLE": "client"}
    result = subprocess.run(
        [
            PYTHON, RUN_LOAD_PY,
            "--workload", workload_path,
            "--router-url", ROUTER_URL,
            "--arrival-gaps-ms", *[str(g) for g in gaps_ms],
            "--reference", str(reference),
            "--output", output_csv,
        ],
        env=load_env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    nvlink_after = nvlink_snapshot()
    Path(output_csv).with_suffix(".nvlink.json").write_text(json.dumps({
        "before_kib": nvlink_before,
        "after_kib": nvlink_after,
        "delta": nvlink_delta(nvlink_before, nvlink_after),
    }, indent=2))
    Path(output_csv).with_suffix(".load.log").write_text(result.stdout + result.stderr)
    rows = list(csv.DictReader(open(output_csv))) if Path(output_csv).exists() else []
    valid = (result.returncode == 0 and len(rows) == 15 and
             all(r.get("success") == "True" and r.get("reference_match") == "True" for r in rows))
    if valid:
        workload_rows = [json.loads(line) for line in Path(workload_path).read_text().splitlines() if line.strip()]
        verification = validate_run(output_csv, workload_rows, read_references(reference))
        valid = verification['valid']
        Path(output_csv).with_suffix('.validation.json').write_text(json.dumps(verification, indent=2))
    Path(output_csv).with_suffix(".run.json").write_text(json.dumps({
        "started_utc": started_utc, "ended_utc": datetime.now().isoformat(),
        "gpu_prefill": P_GPU, "gpu_decode": D_GPU,
        "replay_id": replay_id, "success": valid, "requests": len(rows), "returncode": result.returncode,
    }, indent=2))
    if not valid:
        logger.error("run_load failed:\n%s", result.stderr[-2000:])
        return False
    logger.info("run_load OK → %s", output_csv)
    return True


# ── Experiment orchestration ─────────────────────────────────────────────── #

def snapshot_input(source: Path, destination: Path) -> Path:
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise RuntimeError(f"Existing input snapshot differs from {source}: {destination}")
    else:
        shutil.copyfile(source, destination)
    return destination


def run_experiment(
    result_dir: Path,
    workload: str,
    rounds: int,
    do_warmup: bool,
    dry_run: bool,
    schemes: list[str] | None = None,
    resume: bool = False,
    reference: Path | None = None,
) -> None:
    schemes = schemes or SCHEMES
    reference = reference or reference_path()
    result_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        workload = str(snapshot_input(Path(workload), result_dir / "workload.jsonl"))
        reference = snapshot_input(reference, result_dir / "reference_outputs_a100.csv")
    (result_dir / "runner_pid.txt").write_text(str(os.getpid()))
    progress_path = result_dir / "progress.json"
    summary = json.loads(progress_path.read_text()) if resume and progress_path.exists() else []
    if resume:
        requests = [json.loads(line) for line in Path(workload).read_text().splitlines() if line.strip()]
        references = read_references(reference)
        for entry in summary:
            if not entry["success"] or not validate_run(entry["csv"], requests, references)["valid"]:
                raise RuntimeError(f"Resume requires valid saved runs: {entry['csv']}")
    completed = {(s["round"], s["scheme"], s["scenario"]) for s in summary if s["success"]}

    for round_idx in range(rounds):
        offset = round_idx % len(schemes)
        scheme_order = schemes[offset:] + schemes[:offset]
        scenario_order = ["50ms", "25ms"] if round_idx % 2 else ["25ms", "50ms"]
        logger.info(
            "=== Round %d/%d: schemes=%s scenarios=%s ===",
            round_idx + 1, rounds, scheme_order, scenario_order,
        )

        for scheme in scheme_order:
            if all((round_idx+1, scheme, scenario) in completed for scenario in scenario_order):
                logger.info("Resume: keeping completed round=%d scheme=%s", round_idx+1, scheme)
                continue
            scheme_log_dir = result_dir / f"logs_round{round_idx+1}_{scheme}"
            if resume and scheme_log_dir.exists():
                if any((round_idx+1, scheme, scenario) in completed for scenario in scenario_order):
                    raise RuntimeError("Cannot resume a partially completed service without a separate log directory")
                archived = result_dir / "failed_attempts" / (scheme_log_dir.name + "_" + datetime.now().strftime("%H%M%S"))
                archived.parent.mkdir(exist_ok=True)
                scheme_log_dir.rename(archived)
            scheme_log_dir.mkdir(parents=True, exist_ok=True)

            logger.info("[%s] Launching scheme=%s", ts(), scheme)
            router_proc = None
            try:
                if not dry_run:
                    launch_scheme(scheme, scheme_log_dir)
                    router_proc = start_router(scheme_log_dir)
                for scenario_name in scenario_order:
                    gaps = SCENARIOS[scenario_name]

                    # Warmup
                    if do_warmup:
                        logger.info(
                            "[%s] Warmup: scheme=%s scenario=%s",
                            ts(), scheme, scenario_name,
                        )
                        warmup_csv = str(
                            scheme_log_dir / f"warmup_{scenario_name}.csv"
                        )
                        if not dry_run:
                            if not run_load(workload, gaps, warmup_csv, reference):
                                raise RuntimeError(f"Warmup validation failed: {warmup_csv}")
                        else:
                            logger.info("  [dry-run] skip warmup")

                    # Formal run
                    logger.info(
                        "[%s] Formal: scheme=%s scenario=%s round=%d",
                        ts(), scheme, scenario_name, round_idx + 1,
                    )
                    run_name = (
                        f"{scheme}_{scenario_name}_run{round_idx+1}"
                    )
                    run_dir = result_dir / "runs" / run_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    formal_csv = str(run_dir / "requests.csv")

                    success = False
                    if not dry_run:
                        success = run_load(workload, gaps, formal_csv, reference)
                    else:
                        logger.info("  [dry-run] skip formal")
                        success = True

                    summary.append({
                        "round": round_idx + 1,
                        "scheme": scheme,
                        "scenario": scenario_name,
                        "success": success,
                        "csv": formal_csv,
                        "gpu_prefill": P_GPU,
                        "gpu_decode": D_GPU,
                    })
                    (result_dir / "progress.json").write_text(json.dumps(summary, indent=2))
                    if not success:
                        raise RuntimeError(f"Formal run failed: {run_name}")

            finally:
                if not dry_run:
                    if router_proc:
                        stop_router(router_proc)
                    stop_pair(scheme_log_dir)
                logger.info("[%s] scheme=%s done", ts(), scheme)
                if not dry_run:
                    time.sleep(3)  # Cool-down between schemes

    # Write experiment summary
    summary_path = result_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "rounds": rounds,
            "config": {
                "schemes": {name: {
                    "priority_preemption": name in ("old", "new", "preempt_only"),
                    "local_sort": name in ("new", "sort_only"),
                    "stream_by_batch": name != "baseline",
                } for name in schemes},
                "workload": str(Path(workload).resolve()),
                "reference": str(reference.resolve()),
                "max_tokens": 64,
                "gpu_prefill": next(iter({s.get('gpu_prefill', P_GPU) for s in summary})) if len({s.get('gpu_prefill', P_GPU) for s in summary}) == 1 else None,
                "gpu_decode": next(iter({s.get('gpu_decode', D_GPU) for s in summary})) if len({s.get('gpu_decode', D_GPU) for s in summary}) == 1 else None,
                "gpu_pairs": sorted({f"{s.get('gpu_prefill', P_GPU)}/{s.get('gpu_decode', D_GPU)}" for s in summary}),
                "ucx_tls": "cuda,tcp",
                "ucx_net_devices": "lo",
                "ucx_log_level": "info",
                "local_sort_window_ms": float(os.environ.get("PD_LOCAL_SORT_WINDOW_MS", "10")),
                "batch_start_times_ms": SCENARIOS,
                "warmup": do_warmup,
                "chunk_size_blocks": int(os.environ.get("CHUNK_SIZE_BLOCKS", "8")),
                "stream_max_in_flight": int(os.environ.get("PD_STREAM_MAX_IN_FLIGHT", "1")),
                "stream_poll_ms": float(os.environ.get("PD_STREAM_POLL_MS", "0.1")),
            },
            "runs": summary,
        }, f, indent=2)
    logger.info("Summary → %s", summary_path)

    # Print quick stats
    total = len(summary)
    ok = sum(1 for s in summary if s["success"])
    logger.info("Experiment complete: %d/%d successful runs", ok, total)
    for s in summary:
        status = "OK" if s["success"] else "FAIL"
        logger.info(
            "  [%s] Round%d %s/%s → %s",
            status, s["round"], s["scheme"], s["scenario"], s["csv"],
        )


# ── Entry point ──────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="P/D sorting and preemption comparison experiment"
    )
    parser.add_argument(
        "--result-dir",
        default=str(
            RESULTS_DIR
            / f"nvlink-local-sort-comparison-{datetime.now().strftime('%Y%m%d')}"
        ),
    )
    parser.add_argument(
        "--workload",
        default=str(DEFAULT_WORKLOAD),
    )
    parser.add_argument("--reference", type=Path, default=reference_path(),
                        help="Reference token CSV (default: PD_REFERENCE or the bundled A100 reference)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--schemes", nargs="+", default=SCHEMES,
                        choices=["baseline", "old", "new", "sort_only", "preempt_only", "stream_fifo"])
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Keep successful progress.json entries and run missing services")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Check input files and print the plan without starting services or writing results",
    )
    args = parser.parse_args()

    try:
        check_experiment_inputs(args.workload, args.dry_run, args.reference)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError, RuntimeError) as exc:
        logger.error("Preflight failed: %s", exc)
        sys.exit(1)

    result_dir = Path(args.result_dir)
    if result_dir.exists() and not args.dry_run:
        # Safety: refuse to overwrite existing formal results
        runs_dir = result_dir / "runs"
        if runs_dir.exists() and any(runs_dir.iterdir()) and not args.resume:
            logger.error(
                "Result directory %s already has runs. "
                "Use a different --result-dir.",
                result_dir,
            )
            sys.exit(1)

    logger.info("Result dir: %s", result_dir)
    logger.info("Workload:   %s", args.workload)
    logger.info("Reference:  %s", args.reference)
    logger.info("Rounds:     %d", args.rounds)
    logger.info("Warmup:     %s", args.warmup)
    logger.info("Dry run:    %s", args.dry_run)
    if args.dry_run:
        for round_idx in range(args.rounds):
            offset = round_idx % len(args.schemes)
            order = list(SCENARIOS) if round_idx % 2 == 0 else list(reversed(SCENARIOS))
            logger.info("Round %d: schemes=%s scenarios=%s", round_idx + 1,
                        args.schemes[offset:] + args.schemes[:offset], order)
        return

    run_experiment(
        result_dir=result_dir,
        workload=args.workload,
        rounds=args.rounds,
        do_warmup=args.warmup,
        dry_run=args.dry_run,
        schemes=args.schemes,
        resume=args.resume,
        reference=args.reference,
    )


if __name__ == "__main__":
    main()
