#!/usr/bin/env python3
"""Join diagnostic replays by unique IDs and host monotonic timestamps."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    events = []
    for file in (root / "trace").glob("*.jsonl"):
        for line in file.read_text().splitlines():
            events.append(json.loads(line))
    events.sort(key=lambda e: e["t_ns"])
    mapping = {e["shared"]: (e["replay"], e["rid"])
               for e in events if e["event"] == "router_body"}
    joined = defaultdict(lambda: defaultdict(list))
    steps = defaultdict(list)
    for event in events:
        name, role = event["event"], event["role"]
        if name == "engine_step":
            steps[event["pid"]].append(event)
        key = None
        if event.get("replay") and event.get("rid"):
            key = event["replay"], event["rid"]
        elif event.get("shared"):
            key = mapping.get(event["shared"])
        elif event.get("rid"):
            match = re.search(r"cmpl-([0-9a-f]{16})(?:-|$)", event["rid"])
            key = mapping.get(match[1]) if match else None
        if key:
            stage = event.get("stage", "")
            joined[key][(name, role, stage)].append(event)
        if name == "scheduled":
            for rid, count in event["tokens"].items():
                match = re.search(r"cmpl-([0-9a-f]{16})(?:-|$)", rid)
                key = mapping.get(match[1]) if match else None
                if key:
                    joined[key][(name, role, "")].append(dict(event, rid=rid, count=count))

    records, windows, run_rows = [], [], []
    for path in sorted(root.glob("runs/*/requests.csv")):
        if not path.with_suffix(".run.json").exists() and not (root / "experiment_summary.json").exists():
            continue  # A live replay writes CSV before its validation metadata.
        match = re.fullmatch(r"(.+)_(25ms|50ms)_run(\d+)", path.parent.name)
        scheme, scenario, round_idx = match.groups()
        replay_meta = json.loads(path.with_suffix(".run.json").read_text())
        if not replay_meta["success"]:
            raise ValueError(f"Invalid replay: {path}")
        replay = replay_meta["replay_id"]
        counter_keys = json.loads(path.with_suffix(".nvlink.json").read_text())["delta"]
        p_gpu = replay_meta.get("gpu_prefill")
        d_gpu = replay_meta.get("gpu_decode")
        if p_gpu is None:
            p_gpu = int(next(re.fullmatch(r"gpu(\d+)_tx_gb", k)[1]
                             for k in counter_keys if k.endswith("_tx_gb")))
        if d_gpu is None:
            d_gpu = int(next(re.fullmatch(r"gpu(\d+)_rx_gb", k)[1]
                             for k in counter_keys if k.endswith("_rx_gb")))
        rows = list(csv.DictReader(path.open()))
        window_groups = defaultdict(list)
        for row in rows:
            data = joined[(replay, row["request_id"])]

            def get(name, role, stage="", final=None):
                candidates = data[(name, role, stage)]
                if final is not None:
                    candidates = [e for e in candidates if bool(e.get("final")) == final]
                return candidates[0] if candidates else None

            record = dict(scheme=scheme, scenario=scenario, round=int(round_idx),
                          gpu_prefill=p_gpu, gpu_decode=d_gpu,
                          replay=replay, rid=row["request_id"], category=row["category"],
                          batch=int(row["batch_idx"]), ttft_ms=float(row["ttft_ms"]))
            stages = {
                "client": get("client_send", "client", "client"),
                "client_headers": get("http_headers_sent", "client", "client"),
                "router": get("router_body", "router"),
                "p_headers": get("http_headers_sent", "router", "P"),
                # Production ingress wraps the diagnostic ASGI stack, so its
                # timestamp precedes body buffering and is the real P entry.
                "api": get("ingress_http", "P") or get("api_enter", "P"),
                "api_body": get("api_body", "P"),
                "socket": get("engine_socket", "P"),
                "enqueue": get("engine_enqueue", "P"),
                "dequeue": get("engine_dequeue", "P"),
                "add": get("engine_add", "P"),
                "admit": get("window_admit", "P"),
                "prepared": get("window_admit", "P"),
                "release": get("window_release", "P") or get("window_singleton", "P"),
                "scheduled": get("scheduled", "P"),
                "frontier_final": get("kv_frontier_recorded", "P", final=True),
                "ready_final": get("chunk_ready_observed", "P", final=True),
                "eligible_final": get("chunk_eligible_observed", "P", final=True),
                "write_final": get("write_submit", "P", final=True),
                "done_final": get("write_done_observed", "P", final=True),
                "p_http_done": get("p_http_done", "router"),
                "native_published": get("native_blocks_published", "P"),
                "native_write": get("native_write_call", "P"),
                "native_done": get("native_done_observed", "P"),
                "first_token": get("client_first_token", "client", "client"),
            }
            for name, event in stages.items():
                record[name + "_ns"] = event["t_ns"] if event else None
            if stages["api"] and stages["api"]["event"] == "ingress_http":
                record["api_ns"] = stages["api"]["arrival_ns"]
            release = stages["release"]
            if release:
                record["admission_source"] = release.get("source", "scheduler")
                record["window"] = release.get("window", "singleton")
                record["rank"] = release.get("rank", 0)
                record["release_ns"] = release["released_ns"]
                record["admit_ns"] = release["admitted_ns"]
                record["deadline_ns"] = release.get("deadline_ns")
                window_key = (str(record["window"]) if "window" in release
                              else "singleton:" + record["rid"])
                window_groups[(release["pid"], window_key)].append(record)
            intervals = {
                "client_to_api_ms": ("client", "api"),
                "client_to_router_ms": ("client", "router"),
                "router_to_p_headers_ms": ("router", "p_headers"),
                "p_headers_to_api_ms": ("p_headers", "api"),
                "api_to_socket_ms": ("api", "socket"),
                "input_queue_ms": ("enqueue", "dequeue"),
                "window_wait_ms": ("admit", "release"),
                "ingress_to_prepared_ms": ("api", "prepared"),
                "prepared_to_release_ms": ("prepared", "release"),
                "ready_queue_ms": ("release", "scheduled"),
                "client_to_schedule_ms": ("client", "scheduled"),
                "schedule_to_frontier_ms": ("scheduled", "frontier_final"),
                "frontier_to_done_ms": ("frontier_final", "done_final"),
                "frontier_to_ready_observed_ms": ("frontier_final", "ready_final"),
                "ready_to_eligible_observed_ms": ("ready_final", "eligible_final"),
                "eligible_to_submit_ms": ("eligible_final", "write_final"),
                "final_submit_to_done_ms": ("write_final", "done_final"),
                "done_to_first_ms": ("done_final", "first_token"),
                "schedule_to_p_http_ms": ("scheduled", "p_http_done"),
                "schedule_to_native_publish_ms": ("scheduled", "native_published"),
                "native_publish_to_write_ms": ("native_published", "native_write"),
                "native_write_to_done_ms": ("native_write", "native_done"),
                "native_done_to_first_ms": ("native_done", "first_token"),
            }
            for name, (a, b) in intervals.items():
                record[name] = ((record[b+"_ns"]-record[a+"_ns"])/1e6
                                if record.get(a+"_ns") and record.get(b+"_ns") else None)
            record["deadline_to_release_ms"] = ((record["release_ns"]-record["deadline_ns"])/1e6
                if record.get("deadline_ns") else None)
            if stages["enqueue"]:
                enqueue = stages["enqueue"]["t_ns"]
                blocked = [s for s in steps[stages["enqueue"]["pid"]]
                           if s["start_ns"] <= enqueue <= s["end_ns"]]
                if blocked:
                    step = blocked[0]
                    record["enqueue_during_step_ms"] = (step["end_ns"]-step["start_ns"])/1e6
                    record["step_remaining_at_enqueue_ms"] = (step["end_ns"]-enqueue)/1e6
            record["internal_rid"] = stages["add"]["rid"] if stages["add"] else None
            record["shared"] = stages["router"]["shared"] if stages["router"] else None
            records.append(record)
        for (pid, wid), members in window_groups.items():
            summary = dict(scheme=scheme, scenario=scenario, round=int(round_idx),
                           gpu_prefill=p_gpu, gpu_decode=d_gpu,
                           replay=replay, pid=pid, window=wid, count=len(members),
                           batches="/".join(map(str, sorted({m["batch"] for m in members}))),
                           members="/".join(m["rid"] for m in sorted(members, key=lambda m: m["rank"])))
            for category in ("short", "medium", "long"):
                summary[category] = sum(m["category"] == category for m in members)
            for stage in ("client", "api", "socket", "enqueue", "dequeue", "admit"):
                stamps = [m[stage+"_ns"] for m in members if m[stage+"_ns"]]
                summary[stage+"_span_ms"] = (max(stamps)-min(stamps))/1e6 if stamps else None
            summary["atomic_before_schedule"] = max(m["add_ns"] for m in members) <= min(
                m["scheduled_ns"] for m in members) if all(
                m.get("add_ns") and m.get("scheduled_ns") for m in members) else False
            windows.append(summary)
        group = [r for r in records if r["replay"] == replay]
        stats = dict(scheme=scheme, scenario=scenario, round=int(round_idx), replay=replay,
                     gpu_prefill=p_gpu, gpu_decode=d_gpu)
        for category in ("short", "medium", "long", "all"):
            subset = [r for r in group if category == "all" or r["category"] == category]
            stats[category+"_ttft_ms"] = statistics.mean(r["ttft_ms"] for r in subset)
        stats["windows"] = len(window_groups)
        stats["cross_batch_windows"] = sum(len({r["batch"] for r in members}) > 1
                                             for members in window_groups.values())
        run_rows.append(stats)
    write_csv(root / "pipeline_requests.csv", records)
    write_csv(root / "pipeline_windows.csv", windows)
    write_csv(root / "pipeline_runs.csv", run_rows)
    groups = defaultdict(list)
    for record in records:
        groups[(record["scheme"], record["scenario"], record["category"])].append(record)
        groups[(record["scheme"], record["scenario"], "all")].append(record)
    aggregate = []
    metrics = ["ttft_ms", "client_to_api_ms", "api_to_socket_ms", "input_queue_ms",
               "window_wait_ms", "ready_queue_ms", "client_to_schedule_ms",
               "schedule_to_frontier_ms", "frontier_to_done_ms", "final_submit_to_done_ms", "done_to_first_ms",
               "schedule_to_p_http_ms"]
    metrics += ["frontier_to_ready_observed_ms", "ready_to_eligible_observed_ms", "eligible_to_submit_ms"]
    metrics += ["schedule_to_native_publish_ms", "native_publish_to_write_ms", "native_write_to_done_ms",
                "native_done_to_first_ms"]
    metrics += ["ingress_to_prepared_ms", "prepared_to_release_ms", "deadline_to_release_ms"]
    metrics += ["client_to_router_ms", "router_to_p_headers_ms", "p_headers_to_api_ms"]
    for (scheme, scenario, category), subset in sorted(groups.items()):
        out = dict(scheme=scheme, scenario=scenario, category=category, n=len(subset))
        for metric in metrics:
            values = [r[metric] for r in subset if r.get(metric) is not None]
            out[metric] = statistics.mean(values) if values else None
        aggregate.append(out)
    write_csv(root / "pipeline_aggregate.csv", aggregate)
    device_groups = defaultdict(list)
    for record in records:
        for category in (record["category"], "all"):
            device_groups[(record["gpu_prefill"], record["gpu_decode"], record["scheme"],
                           record["scenario"], category)].append(record)
    device_rows = []
    for (p_gpu, d_gpu, scheme, scenario, category), subset in sorted(device_groups.items()):
        out = dict(gpu_prefill=p_gpu, gpu_decode=d_gpu, scheme=scheme, scenario=scenario,
                   category=category, n=len(subset), replays=len({r["replay"] for r in subset}))
        for metric in metrics:
            values = [r[metric] for r in subset if r.get(metric) is not None]
            out[metric] = statistics.mean(values) if values else None
        device_rows.append(out)
    write_csv(root / "pipeline_by_gpu_pair.csv", device_rows)
    writer_rows = []
    for run in run_rows:
        subset = [r for r in records if r["replay"] == run["replay"]]
        by_id = {r["internal_rid"]: r for r in subset}
        log = root / f"logs_round{run['round']}_{run['scheme']}" / f"p_{run['scheme']}.log"
        writes, dones, bypass = [], [], 0
        for line in log.read_text(errors="replace").splitlines():
            if match := re.search(r"PUSH_CHUNK_WRITE req=(\S+).*submit_ms=([\d.]+) queue_ms=([\d.]+)", line):
                if match[1] in by_id:
                    writes.append(dict(rid=match[1], category=by_id[match[1]]["category"],
                                       submit_ms=float(match[2]), queue_ms=float(match[3])))
            elif match := re.search(r"PUSH_CHUNK_DONE req=(\S+).*observed_ms=([\d.]+)", line):
                if match[1] in by_id:
                    dones.append(dict(rid=match[1], category=by_id[match[1]]["category"],
                                      observed_ms=float(match[2])))
            elif match := re.search(r"Preemption: short req (\S+) bypasses long req (\S+)", line):
                if match[1] in by_id:
                    if match[2] not in by_id:
                        raise ValueError("Cross-replay writer bypass")
                    bypass += 1
        out = {k: run[k] for k in ("scheme", "scenario", "round", "replay")}
        out.update(writes=len(writes), dones=len(dones), bypass_records=bypass)
        for category in ("short", "medium", "long", "all"):
            for source, metric in ((writes, "submit_ms"), (writes, "queue_ms"), (dones, "observed_ms")):
                values = [e[metric] for e in source if category == "all" or e["category"] == category]
                out[category+"_"+metric] = statistics.mean(values) if values else None
        writer_rows.append(out)
    write_csv(root / "writer_formal.csv", writer_rows)
    formal_by_internal = {r["internal_rid"]: r for r in records}
    choice_rows, choice_counts = [], defaultdict(lambda: defaultdict(int))
    for event in events:
        if event["event"] != "writer_choice" or event["rid"] not in formal_by_internal:
            continue
        request = formal_by_internal[event["rid"]]
        candidates = event["eligible"]
        selected = next(c for c in candidates if c["rid"] == event["rid"] and c["offset"] == event["offset"])
        row = {k: request[k] for k in ("scheme", "scenario", "round", "replay", "gpu_prefill", "gpu_decode")}
        row.update(t_ns=event["t_ns"], selected_rid=event["rid"], selected_offset=event["offset"],
                   selected_category=selected["category"], selected_protected=selected["protected"],
                   fifo_rid=candidates[0]["rid"], fifo_category=candidates[0]["category"],
                   candidate_count=len(candidates),
                   protected_long_over_short=selected["category"] == "long" and selected["protected"] and candidates[0]["category"] == "short",
                   short_over_long=selected["category"] == "short" and candidates[0]["category"] == "long")
        choice_rows.append(row)
        counts = choice_counts[request["replay"]]
        counts["decisions_with_short_eligible"] += 1
        counts["protected_long_over_short"] += int(row["protected_long_over_short"])
        counts["short_over_long"] += int(row["short_over_long"])
    write_csv(root / "writer_choices.csv", choice_rows)
    choice_summary = []
    for run in run_rows:
        if run["replay"] in choice_counts:
            choice_summary.append({**run, **choice_counts[run["replay"]]})
    write_csv(root / "writer_choice_summary.csv", choice_summary)
    workload = [json.loads(line) for line in (root / "workload.jsonl").read_text().splitlines() if line.strip()]
    payload = sum(((len(req["prompt_token_ids"])+15)//16)*16*32*2*8*128*2 for req in workload)
    checks = []
    for path in sorted(list(root.glob("runs/*/requests.nvlink.json")) + list(root.glob("logs_round*/warmup_*.nvlink.json"))):
        deltas = json.loads(path.read_text())["delta"]
        passed = all(abs(value*1e9-payload) <= payload*.001 for value in deltas.values())
        checks.append(dict(path=str(path.relative_to(root)), delta=deltas, passed=passed))
    (root / "nvlink_validation.json").write_text(json.dumps(dict(expected_payload_bytes_per_load=payload,
        tolerance_relative=.001, all_passed=all(c["passed"] for c in checks), loads=checks), indent=2))
    errors = []
    required = ["client", "client_headers", "router", "p_headers", "api", "api_body",
                "socket", "enqueue", "dequeue", "add", "scheduled", "p_http_done", "first_token"]
    ordered = ["client", "router", "p_headers", "api", "socket", "enqueue", "dequeue", "add", "scheduled"]
    for record in records:
        rid = record["replay"] + "/" + record["rid"]
        expected = required + (["frontier_final", "write_final", "done_final"]
                               if record["scheme"] != "baseline" else [])
        if record.get("admission_source") == "p_http_ingress":
            # One opaque whole-window UTILITY replaces individual ADD messages.
            expected = [s for s in expected if s not in ("enqueue", "dequeue")]
            if not record["admit_ns"] <= record["api_ns"] < record["deadline_ns"]:
                errors.append(f"{rid}: ingress timestamp outside fixed window")
            if record["release_ns"] < record["deadline_ns"]:
                errors.append(f"{rid}: window released before deadline")
        if record["scheme"] in ("sort_only", "new"):
            expected += ["admit", "release"]
        for stage in expected:
            if record[stage+"_ns"] is None:
                errors.append(f"{rid}: missing {stage}")
        present = [record[stage+"_ns"] for stage in ordered if record[stage+"_ns"]]
        if present != sorted(present):
            errors.append(f"{rid}: timestamps out of pipeline order")
    for window in windows:
        if not window["atomic_before_schedule"]:
            errors.append(f"{window['replay']}/{window['window']}: partial window scheduled")
    stats = dict(formal_runs=len(run_rows), requests=len(records), errors=errors,
                 checks_passed=bool(records) and not errors,
                 complete=(root / "experiment_summary.json").exists())
    (root / "pipeline_validation.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(dict(formal_runs=len(run_rows), requests=len(records), windows=len(windows),
                          missing_add=sum(r["add_ns"] is None for r in records),
                          missing_schedule=sum(r["scheduled_ns"] is None for r in records),
                          cross_batch_windows=sum("/" in w["batches"] for w in windows))))
    return records, windows, run_rows, aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    analyze(parser.parse_args().root)
