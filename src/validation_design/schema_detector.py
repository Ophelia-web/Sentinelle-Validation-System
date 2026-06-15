"""Heuristics for detecting validation-relevant tabular schema structure."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd


TIME_NAME_HINTS = (
    "date",
    "time",
    "timestamp",
    "month",
    "year",
    "week",
    "period",
    "period_id",
    "report_period",
)
GROUP_NAME_HINTS = (
    "user",
    "patient",
    "customer",
    "store",
    "hospital",
    "school",
    "state",
    "county",
    "region",
    "jurisdiction",
    "site",
    "entity",
    "group",
    "id",
)
TARGET_NAME_HINTS = ("target", "label", "outcome", "y", "response", "rate", "score")


def _name_contains(name: str, hints: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(hint in lower for hint in hints)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    sample = series.dropna().astype(str).head(200)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=False)
    return parsed.notna().mean() >= 0.8


def _column_summary(df: pd.DataFrame, col: str) -> dict[str, Any]:
    series = df[col]
    n_rows = len(df)
    n_unique = int(series.nunique(dropna=True))
    return {
        "name": col,
        "dtype": str(series.dtype),
        "missing_pct": _safe_ratio(float(series.isna().sum()), float(n_rows)),
        "n_unique": n_unique,
        "unique_ratio": _safe_ratio(float(n_unique), float(n_rows)),
        "example_values": [str(v) for v in series.dropna().unique()[:5]],
    }


def profile_dataframe(df: pd.DataFrame) -> dict:
    """Return a compact profile used by downstream validation recommendations."""

    column_summaries = [_column_summary(df, col) for col in df.columns]
    likely_targets = []
    for summary in column_summaries:
        col = summary["name"]
        if _name_contains(col, TARGET_NAME_HINTS) and summary["unique_ratio"] < 0.95:
            likely_targets.append(
                {
                    "name": col,
                    "reason": "target-like column name and not a pure row identifier",
                    "score": 0.65,
                }
            )

    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "column_summaries": column_summaries,
        "likely_target_columns": likely_targets,
    }


def detect_time_like_columns(df: pd.DataFrame) -> list[dict]:
    """Detect columns that may encode event, observation, or reporting time."""

    candidates: list[dict] = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        score = 0.0
        reasons = []
        if _name_contains(col, TIME_NAME_HINTS):
            score += 0.45
            reasons.append("time-like column name")
        if _is_datetime_like(series):
            score += 0.45
            reasons.append("datetime-like values")
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = _safe_ratio(float(series.nunique(dropna=True)), float(n_rows))
            lower_name = col.lower()
            if any(h in lower_name for h in ("year", "month", "week", "period")) and unique_ratio <= 0.6:
                score += 0.25
                reasons.append("numeric period-like values")
        if score > 0:
            summary = _column_summary(df, col)
            summary.update({"score": round(min(score, 1.0), 3), "reasons": reasons})
            candidates.append(summary)

    return sorted(candidates, key=lambda item: (-item["score"], item["unique_ratio"], item["name"]))


def detect_group_like_columns(df: pd.DataFrame) -> list[dict]:
    """Detect repeated entity/group columns relevant for grouped CV."""

    candidates: list[dict] = []
    n_rows = len(df)
    nameless_entity_threshold = max(10, int(n_rows**0.5))
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = _safe_ratio(float(n_unique), float(n_rows))
        repeated_ratio = 1.0 - unique_ratio
        score = 0.0
        reasons = []
        has_group_name = _name_contains(col, GROUP_NAME_HINTS)
        could_be_nameless_entity = n_unique >= nameless_entity_threshold
        if has_group_name:
            score += 0.45
            reasons.append("group/entity-like column name")
        if 1 < n_unique < n_rows and repeated_ratio >= 0.1:
            if has_group_name:
                score += 0.3
                reasons.append("values repeat across rows")
            elif could_be_nameless_entity:
                score += 0.25
                reasons.append("many repeated categorical values")
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series) or pd.api.types.is_categorical_dtype(series):
            if (has_group_name or could_be_nameless_entity) and 1 < n_unique <= max(1000, int(n_rows * 0.5)):
                score += 0.15
                reasons.append("categorical repeated values")
        if score >= 0.35 and unique_ratio < 0.98:
            summary = _column_summary(df, col)
            summary.update({"score": round(min(score, 1.0), 3), "reasons": reasons})
            candidates.append(summary)

    return sorted(candidates, key=lambda item: (-item["score"], item["unique_ratio"], item["name"]))


def detect_id_like_columns(df: pd.DataFrame) -> list[dict]:
    """Detect pure or near-pure row identifiers that should rarely be features."""

    candidates: list[dict] = []
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = _safe_ratio(float(n_unique), float(n_rows))
        lower_name = col.lower()
        score = 0.0
        reasons = []
        has_id_name = lower_name.endswith("_id") or lower_name == "id" or "row_id" in lower_name or "id" in lower_name
        if lower_name.endswith("_id") or lower_name == "id" or "row_id" in lower_name:
            score += 0.35
            reasons.append("identifier-like column name")
        elif "id" in lower_name:
            score += 0.2
            reasons.append("name contains id")
        mostly_unique_label = (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        )
        if unique_ratio >= 0.95 and n_unique > 1 and (has_id_name or mostly_unique_label):
            score += 0.55
            reasons.append("mostly unique values")
        if score > 0:
            summary = _column_summary(df, col)
            summary.update({"score": round(min(score, 1.0), 3), "reasons": reasons})
            candidates.append(summary)

    return sorted(candidates, key=lambda item: (-item["score"], -item["unique_ratio"], item["name"]))


def detect_block_like_columns(df: pd.DataFrame, target_col: str | None = None) -> list[dict]:
    """Detect low-cardinality category/block columns useful for stratified reporting."""

    candidates: list[dict] = []
    n_rows = len(df)
    target_col = target_col if target_col in df.columns else None
    for col in df.columns:
        if col == target_col:
            continue
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        unique_ratio = _safe_ratio(float(n_unique), float(n_rows))
        if n_unique <= 1 or unique_ratio > 0.2 or n_unique > 100:
            continue
        if pd.api.types.is_numeric_dtype(series) and not _name_contains(col, ("category", "class", "type", "segment", "block", "group")):
            continue
        if unique_ratio >= 0.95:
            continue
        summary = _column_summary(df, col)
        score = 0.45
        reasons = ["low-cardinality candidate block/category column"]
        if _name_contains(col, ("category", "class", "type", "segment", "block")):
            score += 0.25
            reasons.append("block/category-like column name")
        summary.update({"score": round(min(score, 1.0), 3), "reasons": reasons})
        candidates.append(summary)

    return sorted(candidates, key=lambda item: (-item["score"], item["n_unique"], item["name"]))


def infer_task_type(df: pd.DataFrame, target_col: str | None) -> str:
    """Infer regression/classification/unknown from a candidate target column."""

    if not target_col or target_col not in df.columns:
        return "unknown"

    target = df[target_col].dropna()
    if target.empty:
        return "unknown"

    n_unique = int(target.nunique(dropna=True))
    if pd.api.types.is_bool_dtype(target) or pd.api.types.is_categorical_dtype(target):
        return "classification"
    if pd.api.types.is_object_dtype(target) or pd.api.types.is_string_dtype(target):
        return "classification"
    if pd.api.types.is_numeric_dtype(target):
        if n_unique <= min(20, max(2, int(len(target) * 0.05))):
            return "classification"
        return "regression"
    return "unknown"


def infer_data_structure(df: pd.DataFrame, time_cols, group_cols) -> dict:
    """Infer whether rows are IID, grouped, temporal, panel, or uncertain."""

    n_rows = len(df)
    top_time = time_cols[0]["name"] if time_cols else None
    top_group = group_cols[0]["name"] if group_cols else None
    has_time = bool(time_cols)
    repeated_group = False
    panel_evidence = False

    if top_group and top_group in df.columns:
        group_unique_ratio = _safe_ratio(float(df[top_group].nunique(dropna=True)), float(n_rows))
        repeated_group = 0 < group_unique_ratio < 0.9
    if top_time and top_group and top_time in df.columns and top_group in df.columns:
        combo_counts = df.groupby([top_group, top_time], dropna=False).size()
        groups_per_time = df.groupby(top_time, dropna=False)[top_group].nunique()
        times_per_group = df.groupby(top_group, dropna=False)[top_time].nunique()
        panel_evidence = (
            len(combo_counts) > 0
            and groups_per_time.max() > 1
            and times_per_group.max() > 1
            and repeated_group
        )

    if panel_evidence:
        structure = "panel"
        confidence = 0.85
        reason = "repeated group/entity observations across time-like periods"
    elif has_time:
        structure = "temporal"
        confidence = 0.7
        reason = "time-like columns detected"
    elif repeated_group:
        structure = "grouped"
        confidence = 0.7
        reason = "repeated group/entity columns detected"
    elif n_rows < 50:
        structure = "uncertain"
        confidence = 0.4
        reason = "small dataset limits reliable structure detection"
    else:
        structure = "iid"
        confidence = 0.55
        reason = "no strong group or time structure detected"

    return {
        "structure": structure,
        "confidence": confidence,
        "reason": reason,
        "recommended_time_col": top_time,
        "recommended_group_col": top_group,
    }
