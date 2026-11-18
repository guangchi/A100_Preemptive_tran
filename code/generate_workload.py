#!/usr/bin/env python3
"""Generate the fixed 15-request workload JSONL.

Matches Plan.txt §11.2 exactly:
  8 short  (128 tokens, seeds 1500-1514 at positions 0,2,4,6,8,10,12,14)
  3 medium (1107 tokens, seeds at positions 3,9,13)
  4 long   (10000 tokens, seeds at positions 1,5,7,11)

Each request has pre-determined token-ID prompts (to avoid re-tokenization
differences across services). Tokens are filled from a fixed word pool and
truncated to the exact target length.

Output: workload.sweep-interleaved-8s3m4l.jsonl
SHA256 is computed and printed for verification.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from pd_experiment_paths import DEFAULT_WORKLOAD

# ── Fixed word pool for synthetic prompts ─────────────────────────────────── #
# Using common English words ensures deterministic tokenization.
# This is a synthetic workload, not real data.
WORD_POOL = [
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
    "hello", "world", "system", "model", "language", "compute", "memory",
    "data", "transfer", "network", "performance", "latency", "throughput",
    "request", "response", "server", "client", "protocol", "channel",
    "buffer", "cache", "block", "token", "sequence", "batch", "schedule",
    "priority", "queue", "process", "thread", "async", "stream", "event",
    "prefill", "decode", "attention", "transformer", "layer", "weight",
    "gradient", "optimizer", "learning", "rate", "loss", "accuracy",
    "training", "inference", "generation", "sampling", "temperature",
    "context", "prompt", "completion", "embedding", "vector", "matrix",
    "tensor", "operation", "kernel", "parallel", "concurrent", "dispatch",
]


def generate_token_ids(num_tokens: int, seed: int) -> list[int]:
    """Generate deterministic token IDs of the given length.

    Uses a simple LCG seeded with the request seed to produce stable
    token IDs in the range [1000, 50000) – well within the Llama tokenizer's
    common token range.

    This produces SYNTHETIC token sequences. They are not semantically
    meaningful text, but they exercise the exact same KV cache and transfer
    path as real prompts of the same length.
    """
    # LCG parameters (Numerical Recipes)
    a = 1664525
    c = 1013904223
    m = 2**32
    state = seed
    tokens = []
    for _ in range(num_tokens):
        state = (a * state + c) % m
        # Map to token ID range [1000, 50000)
        token_id = 1000 + (state % 49000)
        tokens.append(token_id)
    return tokens


# ── Workload specification (Plan.txt §11.2) ───────────────────────────────── #
# (position, category, num_tokens, seed, batch_idx)
SPEC = [
    (0,  "short",  128,   1500, 0),
    (1,  "long",   10000, 1501, 0),
    (2,  "short",  128,   1502, 0),
    (3,  "medium", 1107,  1503, 0),
    (4,  "short",  128,   1504, 0),
    (5,  "long",   10000, 1505, 1),
    (6,  "short",  128,   1506, 1),
    (7,  "long",   10000, 1507, 1),
    (8,  "short",  128,   1508, 1),
    (9,  "medium", 1107,  1509, 1),
    (10, "short",  128,   1510, 2),
    (11, "long",   10000, 1511, 2),
    (12, "short",  128,   1512, 2),
    (13, "medium", 1107,  1513, 2),
    (14, "short",  128,   1514, 2),
]


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKLOAD

    requests = []
    for pos, category, num_tokens, seed, batch_idx in SPEC:
        token_ids = generate_token_ids(num_tokens, seed)
        req = {
            "request_id": f"{pos:02d}-{category}-{seed}",
            "category": category,
            "num_prompt_tokens": num_tokens,
            "prompt_token_ids": token_ids,
            "max_tokens": 64,
            "seed": seed,
            "batch_idx": batch_idx,
            "position": pos,
        }
        requests.append(req)

    # Write JSONL
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Compute SHA256
    with open(output_path, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    # Summary
    total_tokens = sum(r["num_prompt_tokens"] for r in requests)
    by_cat = {}
    for r in requests:
        by_cat.setdefault(r["category"], 0)
        by_cat[r["category"]] += 1

    print(f"Generated {len(requests)} requests → {output_path}")
    print(f"  Categories: {by_cat}")
    print(f"  Total input tokens: {total_tokens}")
    print(f"  Output length: {requests[0]['max_tokens']} tokens per request")
    print(f"  SHA256: {sha256}")
    print(f"  Seeds: 1500-1514")


if __name__ == "__main__":
    main()
