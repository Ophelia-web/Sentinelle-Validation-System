"""Validation leakage risk checks for tabular prediction tasks."""

from __future__ import annotations

import pandas as pd


def _names(items: list[dict]) -> list[str]:
    return [item["name"] for item in items if "name" in item]


def _risk(severity: str, risk: str, evidence: str, recommendation: str) -> dict:
    return {
        "severity": severity,
        "risk": risk,
        "evidence": evidence,
        "recommendation": recommendation,
    }


def assess_validation_risks(df, target_col, time_cols, group_cols, id_cols, block_cols) -> list[dict]:
    """Assess common validation risks from schema candidates.

    The checks are intentionally transparent and heuristic-based. They should
    prompt review, not replace domain judgment.
    """

    risks: list[dict] = []
    n_rows = len(df)
    time_names = _names(time_cols)
    group_names = _names(group_cols)
    id_names = _names(id_cols)
    block_names = _names(block_cols)

    for group_col in group_names[:3]:
        if group_col not in df.columns:
            continue
        n_groups = int(df[group_col].nunique(dropna=True))
        repeated = n_groups < n_rows
        if repeated:
            risks.append(
                _risk(
                    "high",
                    "Random KFold may leak repeated groups",
                    f"Column `{group_col}` has {n_groups} unique values across {n_rows} rows.",
                    "Use GroupKFold, GroupShuffleSplit, or a group-aware holdout.",
                )
            )
        if 1 < n_groups < 5:
            risks.append(
                _risk(
                    "medium",
                    "Small group count for GroupKFold",
                    f"Column `{group_col}` has only {n_groups} groups.",
                    "Use fewer folds, leave-one-group-out style checks, or a documented holdout.",
                )
            )

    if time_names:
        risks.append(
            _risk(
                "high",
                "Random KFold may leak future information",
                "Time-like columns detected: " + ", ".join(f"`{name}`" for name in time_names[:5]) + ".",
                "Use forward-chaining, last-period holdout, rolling-origin, or blocked time splits.",
            )
        )

    for id_col in id_names[:5]:
        if id_col not in df.columns:
            continue
        unique_ratio = df[id_col].nunique(dropna=True) / max(n_rows, 1)
        if unique_ratio >= 0.95:
            risks.append(
                _risk(
                    "medium",
                    "Potential row identifier leakage",
                    f"Column `{id_col}` is mostly unique ({unique_ratio:.1%} unique).",
                    "Exclude pure row identifiers from model features unless they have documented meaning.",
                )
            )

    lower_target = target_col.lower() if isinstance(target_col, str) else None
    target_like_terms = ("target", "label", "outcome", "response", "rate", "score")
    for col in df.columns:
        lower = col.lower()
        if target_col and col == target_col:
            continue
        if any(term in lower for term in target_like_terms):
            risks.append(
                _risk(
                    "medium",
                    "Target-like feature name",
                    f"Column `{col}` looks target-related while target is `{target_col}`.",
                    "Verify that this column would be available at prediction time.",
                )
            )
        elif lower_target and lower_target in lower:
            risks.append(
                _risk(
                    "medium",
                    "Feature name contains target name",
                    f"Column `{col}` contains target name `{target_col}`.",
                    "Check for derived target leakage.",
                )
            )

    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            continue
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = n_unique / max(n_rows, 1)
        if n_unique > 50 and unique_ratio > 0.3 and col not in id_names:
            risks.append(
                _risk(
                    "medium",
                    "High-cardinality categorical feature may memorize entities",
                    f"Column `{col}` has {n_unique} unique values across {n_rows} rows.",
                    "Consider group-aware validation and encoding that cannot memorize validation rows.",
                )
            )

    for block_col in block_names[:5]:
        if block_col not in df.columns:
            continue
        counts = df[block_col].value_counts(dropna=False)
        if counts.empty:
            continue
        min_share = counts.min() / max(n_rows, 1)
        if len(counts) > 1 and min_share < 0.05:
            risks.append(
                _risk(
                    "low",
                    "Block imbalance may hide subgroup performance",
                    f"Column `{block_col}` has a smallest block share of {min_share:.1%}.",
                    "Report validation metrics by block/category in addition to the overall metric.",
                )
            )

    if not risks:
        risks.append(
            _risk(
                "low",
                "No obvious validation leakage risk detected",
                "Schema heuristics did not find strong repeated-group, time, or identifier signals.",
                "Still confirm the split with domain knowledge and prediction-time availability.",
            )
        )

    return risks
