#!/usr/bin/env python3
"""Analyze 3-scheme P/D comparison experiment results.

Computes per-category and overall TTFT and E2E latency statistics across
all rounds, and prints a full summary table.

Usage:
  python analyze_pd_local_sort_comparison.py \\
      --result-dir results/nvlink-local-sort-comparison-20261113
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from collections import defaultdict

CATEGORIES = ["short", "medium", "long"]


def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def parse_float(v, default: float = -1.0) -> float:
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def fmt_mean_std(vals: list[float]) -> str:
    if not vals:
        return "    N/A"
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:7.0f}±{std:3.0f}"


def collect_runs(runs_dir: Path) -> dict[tuple[str, str], list[list[dict]]]:
    """Return {(scheme, scenario) -> [rows_per_round, ...]}."""
    all_runs: dict[tuple[str, str], list[list[dict]]] = defaultdict(list)
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parts = run_dir.name.split("_")
        if len(parts) < 3:
            continue
        # e.g. "baseline_250ms_run1"
        scenario_part = parts[-2]
        scheme_part = "_".join(parts[:-2])
        csv_path = run_dir / "requests.csv"
        if not csv_path.exists():
            continue
        all_runs[(scheme_part, scenario_part)].append(load_csv(str(csv_path)))
    return all_runs


def per_round_stats(
    rounds: list[list[dict]],
) -> dict:
    """
    For each round compute per-category means of ttft_ms and e2e_ms,
    plus overall (all categories) means and makespan.
    Returns dict of lists (one value per round) for each metric.
    """
    result: dict[str, list[float]] = defaultdict(list)

    for rows in rounds:
        valid = [r for r in rows if r.get("success") in ("True", "1", True)]

        # makespan: wall-clock from first submit to last completion
        submits = [parse_float(r.get("submitted_at", "0")) for r in valid]
        e2es    = [parse_float(r.get("e2e_ms", "-1"))      for r in valid]
        if submits and all(e > 0 for e in e2es):
            ends = [s + e / 1000.0 for s, e in zip(submits, e2es)]
            result["makespan_ms"].append((max(ends) - min(submits)) * 1000.0)

        # per-category and overall ttft / e2e
        all_ttft, all_e2e = [], []
        for cat in CATEGORIES:
            ttfts = [parse_float(r["ttft_ms"]) for r in valid
                     if r.get("category") == cat and parse_float(r.get("ttft_ms", "-1")) > 0]
            e2es_cat = [parse_float(r["e2e_ms"]) for r in valid
                        if r.get("category") == cat and parse_float(r.get("e2e_ms", "-1")) > 0]
            if ttfts:
                result[f"ttft_{cat}"].append(statistics.mean(ttfts))
                all_ttft.extend(ttfts)
            if e2es_cat:
                result[f"e2e_{cat}"].append(statistics.mean(e2es_cat))
                all_e2e.extend(e2es_cat)

        if all_ttft:
            result["ttft_overall"].append(statistics.mean(all_ttft))
        if all_e2e:
            result["e2e_overall"].append(statistics.mean(all_e2e))

    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    runs_dir = result_dir / "runs"
    if not runs_dir.exists():
        print(f"[error] No runs/ directory in {result_dir}")
        return

    all_runs = collect_runs(runs_dir)
    if not all_runs:
        print("[error] No completed runs found.")
        return

    preferred = ["baseline", "sort_only", "preempt_only", "stream_fifo", "old", "new"]
    found = {scheme for scheme, _ in all_runs}
    schemes = [s for s in preferred if s in found] + sorted(found - set(preferred))
    scenarios = sorted(
        {scenario for _, scenario in all_runs},
        key=lambda scenario: int(scenario.removesuffix("ms")),
    )

    # Pre-compute stats for all (scheme, scenario) pairs
    stats: dict[tuple[str, str], dict] = {}
    for scheme in schemes:
        for scenario in scenarios:
            key = (scheme, scenario)
            rounds = all_runs.get(key, [])
            stats[key] = per_round_stats(rounds)

    # ------------------------------------------------------------------ #
    # Print tables                                                         #
    # ------------------------------------------------------------------ #
    sep = "=" * 86

    print(f"\n{sep}")
    print("P/D NVLink Ablation Results  (mean ± std across 3 rounds)")
    print(sep)

    for scenario in scenarios:
        gap = scenario.replace("ms", "")
        print(f"\n── Scenario: batches at 0 / {gap} / {int(gap)*2} ms ──")

        # ── TTFT table ──────────────────────────────────────────────── #
        print(f"\n  [TTFT — Time to First Token, ms]")
        hdr = f"  {'Scheme':<10} {'Makespan':>9}  {'Short':>10}  {'Medium':>10}  {'Long':>10}  {'Overall':>10}"
        print(hdr)
        print("  " + "-" * 64)
        for scheme in schemes:
            s = stats[(scheme, scenario)]
            ms_s = (f"{statistics.mean(s['makespan_ms'])/1000:.3f}s"
                    if s.get("makespan_ms") else "  N/A  ")
            print(
                f"  {scheme:<10} {ms_s:>9}  "
                f"{fmt_mean_std(s.get('ttft_short',  [])):>10}  "
                f"{fmt_mean_std(s.get('ttft_medium', [])):>10}  "
                f"{fmt_mean_std(s.get('ttft_long',   [])):>10}  "
                f"{fmt_mean_std(s.get('ttft_overall',[])):>10}"
            )

        # ── E2E table ────────────────────────────────────────────────── #
        print(f"\n  [E2E — End-to-End Latency, ms]")
        print(hdr)
        print("  " + "-" * 64)
        for scheme in schemes:
            s = stats[(scheme, scenario)]
            ms_s = (f"{statistics.mean(s['makespan_ms'])/1000:.3f}s"
                    if s.get("makespan_ms") else "  N/A  ")
            print(
                f"  {scheme:<10} {ms_s:>9}  "
                f"{fmt_mean_std(s.get('e2e_short',  [])):>10}  "
                f"{fmt_mean_std(s.get('e2e_medium', [])):>10}  "
                f"{fmt_mean_std(s.get('e2e_long',   [])):>10}  "
                f"{fmt_mean_std(s.get('e2e_overall',[])):>10}"
            )

    # ── Individual long request TTFT ─────────────────────────────────── #
    print(f"\n── Individual long request TTFT (ms, mean across rounds) ──")
    long_req_ids: set[str] = set()
    for rounds in all_runs.values():
        for rows in rounds:
            for r in rows:
                if r.get("category") == "long":
                    long_req_ids.add(r.get("request_id", "?"))

    hdr2 = f"{'Req ID':<26}"
    for scheme in schemes:
        hdr2 += f" {scheme:>12}"
    print(hdr2)
    print("-" * (26 + 13 * len(schemes)))

    for req_id in sorted(long_req_ids):
        for scenario in scenarios:
            row_parts = [f"{req_id+'['+scenario+']':<26}"]
            has_data = False
            for scheme in schemes:
                rounds = all_runs.get((scheme, scenario), [])
                ttfts = []
                for round_rows in rounds:
                    for r in round_rows:
                        if r.get("request_id") == req_id and parse_float(r.get("ttft_ms", "-1")) > 0:
                            ttfts.append(parse_float(r["ttft_ms"]))
                            break
                if ttfts:
                    row_parts.append(f" {statistics.mean(ttfts):>12.1f}")
                    has_data = True
                else:
                    row_parts.append(f" {'N/A':>12}")
            if has_data:
                print("".join(row_parts))

    # ── Correctness ──────────────────────────────────────────────────── #
    print(f"\n── Correctness ──")
    total_ok = total_all = 0
    for rounds in all_runs.values():
        for rows in rounds:
            total_ok  += sum(1 for r in rows if r.get("success") in ("True", "1", True))
            total_all += len(rows)
    print(f"  {total_ok}/{total_all} requests succeeded across all runs")

    # ── Save JSON ────────────────────────────────────────────────────── #
    out_json = result_dir / "analysis.json"
    analysis: dict = {}
    for scheme in schemes:
        analysis[scheme] = {}
        for scenario in scenarios:
            key = (scheme, scenario)
            s = stats[key]
            rounds = all_runs.get(key, [])

            def _ms(k: str) -> dict:
                vals = s.get(k, [])
                if not vals:
                    return {}
                return {
                    "mean": round(statistics.mean(vals), 1),
                    "std":  round(statistics.stdev(vals) if len(vals) > 1 else 0, 1),
                }

            analysis[scheme][scenario] = {
                "num_rounds":     len(rounds),
                "total_requests": sum(len(r) for r in rounds),
                "makespan_ms":    _ms("makespan_ms"),
                "ttft": {
                    "short":   _ms("ttft_short"),
                    "medium":  _ms("ttft_medium"),
                    "long":    _ms("ttft_long"),
                    "overall": _ms("ttft_overall"),
                },
                "e2e": {
                    "short":   _ms("e2e_short"),
                    "medium":  _ms("e2e_medium"),
                    "long":    _ms("e2e_long"),
                    "overall": _ms("e2e_overall"),
                },
            }

    with open(out_json, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nAnalysis saved to {out_json}")


if __name__ == "__main__":
    main()
