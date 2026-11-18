#!/usr/bin/env python3
"""Generate reference outputs on A100 (single GPU, no P/D transfer).

Runs the 15-request workload directly on one GPU with VLLM_BATCH_INVARIANT=1
to get deterministic token IDs. These serve as ground truth for verifying
that P/D KV transfer does not corrupt outputs.

Usage:
  CUDA_VISIBLE_DEVICES=6 VLLM_BATCH_INVARIANT=1 \\
    python generate_reference_outputs.py \\
      --workload workload.sweep-interleaved-8s3m4l.jsonl \\
      --output results/reference_outputs_a100.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from pd_experiment_paths import DEFAULT_MODEL, DEFAULT_WORKLOAD, reference_path

logger = logging.getLogger("gen_ref")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

def load_workload(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_reference(workload_path: Path, output_path: Path, model_path: Path) -> None:
    from vllm import LLM, SamplingParams

    logger.info("Loading model %s", model_path)
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        block_size=16,
        enable_prefix_caching=False,
        trust_remote_code=True,
        # Deterministic: same as experiment
        enforce_eager=False,
    )

    workload = load_workload(workload_path)
    logger.info("Loaded %d requests", len(workload))

    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=64,
        seed=None,  # per-request seed set below
    )

    results = []
    for req in workload:
        prompt_token_ids = req["prompt_token_ids"]
        seed = req.get("seed", 0)
        req_id = req["request_id"]

        sp = SamplingParams(
            temperature=0.0,
            max_tokens=req.get("max_tokens", 64), seed=seed
        )
        outputs = llm.generate(
            prompts={"prompt_token_ids": prompt_token_ids},
            sampling_params=sp,
        )
        output = outputs[0]
        output_text = output.outputs[0].text
        output_token_ids = list(output.outputs[0].token_ids)
        token_hash = hashlib.sha256(
            json.dumps(output_token_ids).encode()
        ).hexdigest()[:16]

        results.append({
            "request_id": req_id,
            "category": req.get("category"),
            "num_prompt_tokens": len(prompt_token_ids),
            "num_output_tokens": len(output_token_ids),
            "seed": seed,
            "output_text": output_text[:200],
            "output_token_ids": json.dumps(output_token_ids),
            "token_hash": token_hash,
        })
        logger.info("  %s → %s (hash=%s)", req_id, repr(output_text), token_hash)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    logger.info("Reference outputs saved to %s", output_path)
    logger.info(
        "Run with VLLM_BATCH_INVARIANT=%s",
        os.environ.get("VLLM_BATCH_INVARIANT", "not set"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output", type=Path, default=reference_path())
    parser.add_argument("--model", type=Path, default=os.environ.get("MODEL_PATH") or DEFAULT_MODEL)
    args = parser.parse_args()
    run_reference(args.workload, args.output, args.model.expanduser().resolve())


if __name__ == "__main__":
    main()
