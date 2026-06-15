"""Validation strategy recommendation rules."""

from __future__ import annotations


def _top_names(items: list[dict], limit: int = 3) -> list[str]:
    return [item["name"] for item in items[:limit] if "name" in item]


def recommend_validation_strategy(profile: dict) -> dict:
    """Recommend primary and secondary validation strategies from a profile.

    Expected profile keys include task_type, data_structure, time_cols,
    group_cols, block_cols, and risks. Extra keys are preserved by callers.
    """

    task_type = profile.get("task_type", "unknown")
    data_structure = profile.get("data_structure", {})
    structure = data_structure.get("structure", "uncertain")
    time_cols = profile.get("time_cols", [])
    group_cols = profile.get("group_cols", [])
    block_cols = profile.get("block_cols", [])
    recommended_time_col = data_structure.get("recommended_time_col") or (_top_names(time_cols, 1) or [None])[0]
    recommended_group_col = data_structure.get("recommended_group_col") or (_top_names(group_cols, 1) or [None])[0]
    recommended_block_cols = _top_names(block_cols, 3)

    if structure == "iid":
        if task_type == "classification":
            primary = "StratifiedKFold"
            secondary = ["RepeatedStratifiedKFold", "train/validation holdout stratified by target"]
            explanation = "No strong group or time structure was detected, and the target appears categorical."
        else:
            primary = "KFold or RepeatedKFold"
            secondary = ["ShuffleSplit", "repeated random holdout"]
            explanation = "No strong group or time structure was detected, so IID-style validation is a reasonable starting point."
        avoid = ["using row identifiers as features", "single split without sensitivity checks"]
        risk_level = "low"
    elif structure == "grouped":
        primary = "GroupKFold"
        secondary = ["GroupShuffleSplit", "leave-one-group-out when group count is small enough"]
        avoid = ["shuffled random KFold", "random train/test split that separates the same group across folds"]
        risk_level = "high"
        explanation = "Repeated group/entity values were detected, so validation folds should keep groups intact."
    elif structure == "temporal":
        primary = "TimeSeriesSplit or forward-chaining split"
        secondary = ["last-period holdout", "rolling-origin evaluation"]
        avoid = ["shuffled random KFold", "random split that trains on future rows and validates on past rows"]
        risk_level = "high"
        explanation = "Time-like structure was detected, so validation should respect chronological order."
    elif structure == "panel":
        primary = "blocked_time_split"
        secondary = ["leave-one-period-out", "rolling-origin panel split", "GroupKFold by entity for robustness"]
        avoid = ["shuffled random KFold", "random row split", "time split that ignores repeated entities without reporting block sensitivity"]
        risk_level = "high"
        explanation = "Repeated entities across time-like periods were detected, so panel-aware validation is recommended."
    else:
        primary = "compare_multiple_split_designs"
        secondary = ["KFold baseline", "GroupKFold if a group column is credible", "time holdout if a time column is credible"]
        avoid = ["assuming IID without checking sensitivity"]
        risk_level = "medium"
        explanation = "The data structure is uncertain; compare plausible split designs and report sensitivity."

    if recommended_block_cols:
        secondary = list(secondary) + ["report metrics by block/category columns"]

    return {
        "detected_structure": structure,
        "primary_cv": primary,
        "secondary_cv": secondary,
        "avoid": avoid,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommended_group_col": recommended_group_col,
        "recommended_time_col": recommended_time_col,
        "recommended_block_cols": recommended_block_cols,
    }
