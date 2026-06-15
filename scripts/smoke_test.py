#!/usr/bin/env python
"""Smoke test for Sentinelle Validation Agent benchmark outputs."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    _run([sys.executable, "examples/validation_design_demo/make_mock_panel_data.py"])

    for requested_n_splits in (3, 5):
        output_dir = REPO_ROOT / "artifacts" / f"validation_design_smoke_v21_n{requested_n_splits}"
        _run(
            [
                sys.executable,
                "scripts/validation_design_agent.py",
                "--train",
                "examples/validation_design_demo/mock_panel_data.csv",
                "--target",
                "target",
                "--output-dir",
                str(output_dir),
                "--run-cv-benchmark",
                "--model-name",
                "random_forest",
                "--n-splits",
                str(requested_n_splits),
            ]
        )

        expected_files = [
            output_dir / "validation_design_report.md",
            output_dir / "validation_design_plan.json",
            output_dir / "split_code.py",
            output_dir / "cv_benchmark_results.csv",
        ]
        missing = [path for path in expected_files if not path.exists()]
        if missing:
            raise AssertionError("Missing expected output files: " + ", ".join(str(path) for path in missing))

        plan = json.loads((output_dir / "validation_design_plan.json").read_text(encoding="utf-8"))
        if "cv_benchmark" not in plan:
            raise AssertionError("validation_design_plan.json does not contain cv_benchmark")
        if not plan["cv_benchmark"].get("results"):
            raise AssertionError("cv_benchmark did not contain any evaluated split results")

        with (output_dir / "cv_benchmark_results.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise AssertionError("cv_benchmark_results.csv did not contain any result rows")
        required_columns = {"requested_n_splits", "actual_n_folds"}
        missing_columns = required_columns.difference(rows[0])
        if missing_columns:
            raise AssertionError("Missing benchmark CSV columns: " + ", ".join(sorted(missing_columns)))
        if not any(int(row["actual_n_folds"]) <= int(row["requested_n_splits"]) for row in rows):
            raise AssertionError("Expected at least one design with actual_n_folds <= requested_n_splits")

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
