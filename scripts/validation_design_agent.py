#!/usr/bin/env python
"""CLI for Sentinelle Validation Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation_design import (  # noqa: E402
    assess_validation_risks,
    detect_block_like_columns,
    detect_group_like_columns,
    detect_id_like_columns,
    detect_time_like_columns,
    evaluate_cv_designs,
    generate_split_code,
    infer_data_structure,
    infer_task_type,
    profile_dataframe,
    recommend_validation_strategy,
    write_json_plan,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design leakage-safe validation splits for a tabular prediction dataset."
    )
    parser.add_argument("--train", help="Path to training CSV.")
    parser.add_argument("--target", help="Optional target column name.")
    parser.add_argument("--data-description", help="Optional path to a data description file.")
    parser.add_argument("--output-dir", default="artifacts/validation_design", help="Directory for generated artifacts.")
    parser.add_argument("--max-rows", type=int, default=200000, help="Maximum rows to profile.")
    parser.add_argument("--demo", action="store_true", help="Run on generated mock data if --train is omitted.")
    parser.add_argument("--run-cv-benchmark", action="store_true", help="Train baseline models under candidate CV designs.")
    parser.add_argument("--model-name", default="hist_gradient_boosting", help="Baseline model for --run-cv-benchmark.")
    parser.add_argument("--n-splits", type=int, default=5, help="Maximum CV folds/splits for the CV strategy comparison.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splits, sampling, and models.")
    parser.add_argument(
        "--benchmark-max-rows",
        type=int,
        default=50000,
        help="Maximum rows used by the optional CV strategy comparison.",
    )
    return parser.parse_args()


def _load_demo_data() -> pd.DataFrame:
    demo_dir = REPO_ROOT / "examples" / "validation_design_demo"
    if str(demo_dir) not in sys.path:
        sys.path.insert(0, str(demo_dir))
    from make_mock_panel_data import make_mock_panel_data  # noqa: E402

    return make_mock_panel_data()


def _load_csv(path: str, max_rows: int) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if max_rows <= 0:
        raise ValueError("--max-rows must be a positive integer.")
    return pd.read_csv(csv_path, nrows=max_rows)


def build_plan(df: pd.DataFrame, target_col: str | None, data_description: str | None = None) -> dict:
    if target_col and target_col not in df.columns:
        raise ValueError(
            f"Target column {target_col!r} was not found. Available columns: {', '.join(map(str, df.columns))}"
        )

    dataframe_profile = profile_dataframe(df)
    time_cols = detect_time_like_columns(df)
    group_cols = detect_group_like_columns(df)
    id_cols = detect_id_like_columns(df)
    block_cols = detect_block_like_columns(df, target_col=target_col)
    task_type = infer_task_type(df, target_col)
    data_structure = infer_data_structure(df, time_cols=time_cols, group_cols=group_cols)
    risks = assess_validation_risks(
        df,
        target_col=target_col,
        time_cols=time_cols,
        group_cols=group_cols,
        id_cols=id_cols,
        block_cols=block_cols,
    )

    profile = {
        "dataframe": dataframe_profile,
        "target_col": target_col,
        "task_type": task_type,
        "data_structure": data_structure,
        "time_cols": time_cols,
        "group_cols": group_cols,
        "id_cols": id_cols,
        "block_cols": block_cols,
        "risks": risks,
    }
    recommendation = recommend_validation_strategy(profile)

    return {
        **profile,
        "recommendation": recommendation,
        "data_description": data_description,
    }


def _write_cv_benchmark_csv(benchmark: dict, output_path: Path) -> None:
    rows = []
    for split_name, result in benchmark.get("results", {}).items():
        rows.append(
            {
                "validation_design": split_name,
                "mean_score": result.get("mean_score"),
                "std_score": result.get("std_score"),
                "n_folds": result.get("n_folds"),
                "requested_n_splits": result.get("requested_n_splits"),
                "actual_n_folds": result.get("actual_n_folds"),
                "model_name": result.get("model_name"),
                "metric_name": result.get("metric_name"),
                "interpretation": result.get("interpretation"),
                "adjustment_note": result.get("adjustment_note"),
                "lower_is_better": result.get("lower_is_better"),
                "risk": result.get("risk"),
                "description": result.get("description"),
            }
        )
    pd.DataFrame(
        rows,
        columns=[
            "validation_design",
            "mean_score",
            "std_score",
            "n_folds",
            "requested_n_splits",
            "actual_n_folds",
            "model_name",
            "metric_name",
            "interpretation",
            "adjustment_note",
            "lower_is_better",
            "risk",
            "description",
        ],
    ).to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()
    try:
        if args.train:
            df = _load_csv(args.train, max_rows=args.max_rows)
            source = args.train
        elif args.demo:
            df = _load_demo_data()
            source = "generated demo data"
        else:
            raise ValueError("Provide --train path/to/train.csv or use --demo.")

        data_description_text = None
        if args.data_description:
            desc_path = Path(args.data_description)
            if not desc_path.exists():
                raise FileNotFoundError(f"Data description file not found: {desc_path}")
            data_description_text = desc_path.read_text(encoding="utf-8")

        plan = build_plan(df, target_col=args.target, data_description=data_description_text)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_ran = False
        if args.run_cv_benchmark:
            if not args.target:
                print("Warning: --run-cv-benchmark was supplied without --target; skipping CV strategy comparison.")
            elif plan.get("task_type") not in {"regression", "classification"}:
                plan["cv_benchmark"] = {
                    "enabled": True,
                    "target_col": args.target,
                    "task_type": plan.get("task_type"),
                    "model_name": args.model_name,
                    "metric_name": "unknown",
                    "results": {},
                    "skipped": [
                        {
                            "split": "cv_benchmark",
                            "reason": f"unsupported or unknown task type: {plan.get('task_type')}",
                        }
                    ],
                }
                benchmark_ran = True
            else:
                plan["cv_benchmark"] = evaluate_cv_designs(
                    df,
                    target_col=args.target,
                    plan=plan,
                    task_type=plan["task_type"],
                    model_name=args.model_name,
                    n_splits=args.n_splits,
                    random_state=args.random_state,
                    max_rows=args.benchmark_max_rows,
                )
                benchmark_ran = True

            if benchmark_ran:
                _write_cv_benchmark_csv(plan.get("cv_benchmark", {}), output_dir / "cv_benchmark_results.csv")

        write_markdown_report(plan, str(output_dir / "validation_design_report.md"))
        write_json_plan(plan, str(output_dir / "validation_design_plan.json"))
        (output_dir / "split_code.py").write_text(generate_split_code(plan), encoding="utf-8")

        rec = plan["recommendation"]
        print("Sentinelle Validation Agent")
        print(f"Source: {source}")
        print(f"Rows profiled: {len(df):,}")
        print(f"Detected task type: {plan['task_type']}")
        print(f"Detected structure: {rec['detected_structure']}")
        print(f"Primary CV: {rec['primary_cv']}")
        print(f"Risk level: {rec['risk_level']}")
        if benchmark_ran:
            benchmark = plan.get("cv_benchmark", {})
            result_count = len(benchmark.get("results", {}))
            print(f"CV strategy comparison designs evaluated: {result_count}")
            warning = benchmark.get("optimism_warning")
            if warning and warning.get("detected"):
                print(f"CV strategy comparison warning: {warning.get('message')}")
        print(f"Artifacts written to: {output_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
