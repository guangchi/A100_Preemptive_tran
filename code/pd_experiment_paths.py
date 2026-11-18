"""Defaults shared by the experiment and its input generators."""

import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
REPO_DIR = CODE_DIR.parent
RESULTS_DIR = REPO_DIR / "results"
DEFAULT_WORKLOAD = CODE_DIR / "workload.sweep-interleaved-8s3m4l.jsonl"
DEFAULT_REFERENCE = RESULTS_DIR / "reference_outputs_a100.csv"
DEFAULT_MODEL = REPO_DIR / "models" / "Llama-3.1-8B-Instruct"


def reference_path() -> Path:
    return Path(os.environ.get("PD_REFERENCE") or DEFAULT_REFERENCE).expanduser()
