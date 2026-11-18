"""Resume must preserve completed data and archive failed service startup logs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_pd_local_sort_comparison as runner


class ResumeTests(unittest.TestCase):
    def test_resume_keeps_completed_services_and_archives_failed_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "workload.jsonl"
            workload.write_text('{"request_id":"r","max_tokens":64}\n')
            saved = root / "completed.csv"
            saved.write_text("keep existing output")
            prior = [dict(round=1, scheme="baseline", scenario=scenario,
                          success=True, csv=str(saved)) for scenario in ("25ms", "50ms")]
            (root / "progress.json").write_text(json.dumps(prior))
            failed = root / "logs_round1_sort_only"
            failed.mkdir()
            (failed / "launch.log").write_text("failed startup evidence")
            with patch.object(runner, "validate_run", return_value={"valid": True}), \
                    patch.object(runner, "launch_scheme") as launch:
                runner.run_experiment(root, str(workload), 1, False, True,
                                      schemes=["baseline", "sort_only"], resume=True)
            launch.assert_not_called()
            self.assertEqual(saved.read_text(), "keep existing output")
            summary = json.loads((root / "experiment_summary.json").read_text())
            self.assertEqual(summary["runs"][:2], prior)
            self.assertEqual(len(summary["runs"]), 4)
            self.assertFalse((root / "logs_round1_baseline").exists())
            archives = list((root / "failed_attempts").glob("*/launch.log"))
            self.assertEqual(len(archives), 1)
            self.assertEqual(archives[0].read_text(), "failed startup evidence")

    def test_resume_rejects_invalid_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workload = root / "workload.jsonl"
            workload.write_text('{"request_id":"r","max_tokens":64}\n')
            (root / "progress.json").write_text(json.dumps([
                dict(round=1, scheme="baseline", scenario="25ms", success=True, csv="invalid")]))
            with patch.object(runner, "validate_run", return_value={"valid": False}), \
                    self.assertRaisesRegex(RuntimeError, "valid saved runs"):
                runner.run_experiment(root, str(workload), 1, False, True, resume=True)


if __name__ == "__main__":
    unittest.main()
