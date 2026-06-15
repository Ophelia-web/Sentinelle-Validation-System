"""Empirical cross-validation benchmarks for validation-design sensitivity."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SAFE_SPLIT_PRIORITY = (
    "panel_time_split",
    "time_blocked_split",
    "leave_one_period_out",
    "group_kfold",
    "group_shuffle_split",
)


def _make_one_hot_encoder(sparse: bool = True) -> OneHotEncoder:
    """Return a OneHotEncoder compatible with older and newer sklearn."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=sparse)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=sparse)


def _build_preprocessor(
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    *,
    sparse_output: bool = True,
) -> ColumnTransformer:
    numeric_selector: Any = numeric_cols if numeric_cols is not None else make_column_selector(dtype_include=np.number)
    categorical_selector: Any = (
        categorical_cols
        if categorical_cols is not None
        else make_column_selector(dtype_include=["object", "category", "bool"])
    )

    transformers: list[tuple[str, Pipeline, Any]] = []
    if numeric_cols is None or numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_selector,
            )
        )
    if categorical_cols is None or categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder(sparse=sparse_output)),
                    ]
                ),
                categorical_selector,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False
    sample = series.dropna().astype(str).head(250)
    if sample.empty:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
    return bool(parsed.notna().mean() >= 0.8)


def _datetime_to_numeric(series: pd.Series) -> pd.Series | None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if parsed.notna().mean() < 0.8:
        return None
    values = pd.Series(np.nan, index=series.index, dtype="float64")
    valid = parsed.notna()
    if valid.any():
        values.loc[valid] = parsed.loc[valid].astype("int64").astype("float64") / 1_000_000_000.0
    return values


def _names(items: list[dict] | None) -> list[str]:
    return [item["name"] for item in items or [] if isinstance(item, dict) and "name" in item]


def _recommended_column(plan: dict, key: str, fallback_key: str) -> str | None:
    recommendation = plan.get("recommendation", {})
    data_structure = plan.get("data_structure", {})
    value = recommendation.get(key) or data_structure.get(key)
    if value:
        return value
    candidates = _names(plan.get(fallback_key))
    return candidates[0] if candidates else None


def prepare_modeling_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer, dict]:
    """Prepare features, target, preprocessing, and feature metadata."""

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} was not found.")

    exclude = set(exclude_cols or [])
    exclude.add(target_col)

    y = df[target_col].copy()
    X = df.drop(columns=[col for col in exclude if col in df.columns]).copy()

    datetime_features: list[str] = []
    dropped_features: list[str] = [col for col in exclude if col != target_col and col in df.columns]
    for col in list(X.columns):
        series = X[col]
        if pd.api.types.is_datetime64_any_dtype(series) or _is_datetime_like(series):
            converted = _datetime_to_numeric(series)
            if converted is None:
                X = X.drop(columns=[col])
                dropped_features.append(col)
            else:
                X[col] = converted
                datetime_features.append(col)
        elif pd.api.types.is_bool_dtype(series):
            X[col] = series.astype("float64")

    usable_cols: list[str] = []
    for col in X.columns:
        series = X[col]
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            usable_cols.append(col)
        else:
            dropped_features.append(col)

    X = X[usable_cols]
    numeric_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    categorical_cols = [
        col
        for col in X.columns
        if pd.api.types.is_object_dtype(X[col])
        or pd.api.types.is_string_dtype(X[col])
        or pd.api.types.is_categorical_dtype(X[col])
    ]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable modeling features remain after exclusions and preprocessing.")

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols, sparse_output=True)
    feature_info = {
        "target_col": target_col,
        "exclude_cols": sorted(exclude - {target_col}),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "datetime_features_converted": datetime_features,
        "dropped_features": sorted(set(dropped_features)),
        "n_rows": int(len(X)),
        "n_features_before_encoding": int(len(X.columns)),
    }
    return X, y, preprocessor, feature_info


def _estimator_for(task_type: str, model_name: str, random_state: int):
    task = (task_type or "regression").lower()
    model = (model_name or "hist_gradient_boosting").lower()

    if task == "classification":
        if model == "dummy":
            return DummyClassifier(strategy="most_frequent")
        if model == "logistic_regression":
            return LogisticRegression(max_iter=1000)
        if model == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        if model == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(random_state=random_state)
        raise ValueError(f"Unsupported classification model_name: {model_name!r}")

    if model == "dummy":
        return DummyRegressor(strategy="median")
    if model == "ridge":
        return Ridge()
    if model == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    if model == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=random_state)
    raise ValueError(f"Unsupported regression model_name: {model_name!r}")


def build_baseline_model(
    task_type: str = "regression",
    model_name: str = "hist_gradient_boosting",
    random_state: int = 42,
) -> Pipeline:
    """Return a lightweight sklearn Pipeline with preprocessing and estimator."""

    dense_needed = (model_name or "").lower() == "hist_gradient_boosting"
    preprocessor = _build_preprocessor(sparse_output=not dense_needed)
    estimator = _estimator_for(task_type, model_name, random_state)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _build_model_with_preprocessor(
    preprocessor: ColumnTransformer,
    task_type: str,
    model_name: str,
    random_state: int,
) -> Pipeline:
    estimator = _estimator_for(task_type, model_name, random_state)
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def _requested_split_count(n_splits: int) -> int:
    """Normalize the requested split count as a positive maximum."""

    return max(1, int(n_splits))


def _sort_periods(series: pd.Series) -> list[Any]:
    non_missing = series.dropna()
    if non_missing.empty:
        return []
    unique_values = pd.Series(non_missing.unique())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(unique_values.astype(str), errors="coerce", utc=True)
    if parsed.notna().mean() >= 0.8:
        order = np.argsort(parsed.fillna(pd.Timestamp.max.tz_localize("UTC")).astype("int64").to_numpy())
        return unique_values.iloc[order].tolist()
    try:
        return unique_values.sort_values(kind="mergesort").tolist()
    except TypeError:
        return unique_values.astype(str).sort_values(kind="mergesort").tolist()


def _periods_for_recent_selection(series: pd.Series) -> tuple[list[Any], str]:
    """Return periods ordered for "most recent" selection when possible.

    If values are not directly sortable, keep the original unique-value order so
    the "last" periods are the last observed unique values rather than an
    arbitrary string coercion.
    """

    non_missing = series.dropna()
    if non_missing.empty:
        return [], "no non-missing periods are available"

    unique_values = pd.Series(non_missing.unique())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = pd.to_datetime(unique_values.astype(str), errors="coerce", utc=True)
    if parsed.notna().mean() >= 0.8:
        order = np.argsort(parsed.fillna(pd.Timestamp.max.tz_localize("UTC")).astype("int64").to_numpy())
        return unique_values.iloc[order].tolist(), "most recent sortable periods"
    try:
        return unique_values.sort_values(kind="mergesort").tolist(), "most recent sortable periods"
    except TypeError:
        return unique_values.tolist(), "last observed unique periods because period values were not sortable"


def _split_from_period_blocks(df: pd.DataFrame, time_col: str, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    periods = _sort_periods(df[time_col])
    if len(periods) < 2:
        return []

    valid_periods = periods[1:]
    fold_count = min(max(1, n_splits), len(valid_periods))
    period_blocks = np.array_split(np.array(valid_periods, dtype=object), fold_count)
    indices = np.arange(len(df))
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in period_blocks:
        block_values = list(block)
        if not block_values:
            continue
        first_valid_pos = periods.index(block_values[0])
        train_periods = set(periods[:first_valid_pos])
        valid_periods_set = set(block_values)
        train_mask = df[time_col].isin(train_periods).to_numpy()
        valid_mask = df[time_col].isin(valid_periods_set).to_numpy()
        train_idx = indices[train_mask]
        valid_idx = indices[valid_mask]
        if len(train_idx) and len(valid_idx):
            splits.append((train_idx, valid_idx))
    return splits[:n_splits]


def _make_stratification_labels(y: pd.Series, n_splits: int) -> tuple[pd.Series | None, int | None, str | None]:
    if y is None or y.empty:
        return None, None, "target values are unavailable"
    y_non_null = y.dropna()
    if y_non_null.empty:
        return None, None, "target values are all missing"

    n_unique = int(y_non_null.nunique(dropna=True))
    if n_unique < 2:
        return None, None, "target has fewer than two classes/bins"

    is_classification_like = (
        pd.api.types.is_bool_dtype(y_non_null)
        or pd.api.types.is_object_dtype(y_non_null)
        or pd.api.types.is_string_dtype(y_non_null)
        or pd.api.types.is_categorical_dtype(y_non_null)
        or n_unique <= min(20, max(2, int(len(y_non_null) * 0.05)))
    )
    labels = y.astype(str) if is_classification_like else None
    if labels is None:
        bins = min(10, n_unique, max(2, n_splits))
        try:
            labels = pd.qcut(y.rank(method="first"), q=bins, duplicates="drop").astype(str)
        except ValueError as exc:
            return None, None, f"could not bin regression target for stratification: {exc}"

    counts = labels.value_counts(dropna=False)
    min_count = int(counts.min()) if not counts.empty else 0
    if min_count < 2:
        return None, min_count, "at least one class/bin has fewer than two rows"
    return labels, min_count, None


def generate_candidate_splits(
    df: pd.DataFrame,
    plan: dict,
    y: pd.Series | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Create candidate validation split designs from the detected structure."""

    results: dict[str, dict] = {}
    skipped: list[dict[str, str]] = []
    n_rows = len(df)
    requested_n_splits = _requested_split_count(n_splits)

    def add_skip(name: str, reason: str) -> None:
        skipped.append({"split": name, "reason": reason})

    def add_design(
        name: str,
        splits: list[tuple[np.ndarray, np.ndarray]],
        description: str,
        risk: str,
        adjustment_reason: str | None = None,
    ) -> None:
        valid_splits = [(np.asarray(train), np.asarray(valid)) for train, valid in splits if len(train) and len(valid)]
        valid_splits = valid_splits[:requested_n_splits]
        if valid_splits:
            actual_n_folds = int(len(valid_splits))
            result = {
                "splits": valid_splits,
                "requested_n_splits": int(requested_n_splits),
                "actual_n_folds": actual_n_folds,
                "description": description,
                "risk": risk,
            }
            if actual_n_folds != requested_n_splits:
                reason = adjustment_reason or "only this many valid non-empty folds could be constructed"
                result["adjustment_note"] = (
                    f"Requested up to {requested_n_splits} folds; using {actual_n_folds} because {reason}."
                )
            results[name] = result
        else:
            add_skip(name, "no non-empty train/validation folds could be constructed")

    random_k = min(requested_n_splits, n_rows)
    if random_k >= 2:
        splitter = KFold(n_splits=random_k, shuffle=True, random_state=random_state)
        add_design(
            "random_kfold",
            list(splitter.split(df)),
            "Shuffled row-wise KFold used as an IID diagnostic baseline.",
            "high when repeated groups or time structure are present",
            adjustment_reason=f"only {n_rows} rows are available",
        )
    else:
        add_skip("random_kfold", "at least two requested folds and two rows are required")

    labels, strat_min_count, strat_reason = (
        _make_stratification_labels(y, requested_n_splits) if y is not None and requested_n_splits >= 2 else (None, None, None)
    )
    if labels is not None:
        strat_k = min(requested_n_splits, int(strat_min_count or 0))
        if strat_k >= 2:
            splitter = StratifiedKFold(n_splits=strat_k, shuffle=True, random_state=random_state)
            add_design(
                "stratified_kfold",
                list(splitter.split(df, labels)),
                "Stratified shuffled KFold over classes or binned regression target.",
                "still row-wise; may leak groups or future periods",
                adjustment_reason=f"the smallest class/bin has {strat_min_count} rows",
            )
        else:
            add_skip("stratified_kfold", "not enough rows per class/bin for two folds")
    else:
        add_skip("stratified_kfold", strat_reason or "target was not provided or fewer than two folds were requested")

    group_col = _recommended_column(plan, "recommended_group_col", "group_cols")
    if group_col and group_col in df.columns:
        groups = df[group_col].fillna("__missing_group__")
        n_groups = int(groups.nunique(dropna=False))
        group_k = min(requested_n_splits, n_groups)
        if group_k >= 2:
            splitter = GroupKFold(n_splits=group_k)
            add_design(
                "group_kfold",
                list(splitter.split(df, groups=groups)),
                f"GroupKFold that keeps `{group_col}` values intact across folds.",
                "safer for repeated entities; ignores chronological order",
                adjustment_reason=f"`{group_col}` has {n_groups} unique groups",
            )
        else:
            add_skip(
                "group_kfold",
                f"`{group_col}` has {n_groups} unique groups; at least two groups are required",
            )

        group_shuffle_k = min(requested_n_splits, n_groups)
        if group_shuffle_k >= 2:
            test_size = min(0.5, max(0.2, 1.0 / n_groups))
            splitter = GroupShuffleSplit(
                n_splits=group_shuffle_k,
                test_size=test_size,
                random_state=random_state,
            )
            add_design(
                "group_shuffle_split",
                list(splitter.split(df, groups=groups)),
                f"Repeated group holdout splits based on `{group_col}`.",
                "safer for repeated entities; random over groups",
                adjustment_reason=f"`{group_col}` has {n_groups} unique groups",
            )
        else:
            add_skip("group_shuffle_split", f"`{group_col}` has fewer than two groups")
    else:
        add_skip("group_kfold", "no usable group/entity column was detected")
        add_skip("group_shuffle_split", "no usable group/entity column was detected")

    time_col = _recommended_column(plan, "recommended_time_col", "time_cols")
    if time_col and time_col in df.columns:
        periods = _sort_periods(df[time_col])
        if len(periods) >= 2:
            recent_periods, recent_period_note = _periods_for_recent_selection(df[time_col])
            selected_count = min(requested_n_splits, len(recent_periods))
            selected_periods = recent_periods[-selected_count:]
            indices = np.arange(len(df))
            lopo_splits = []
            for period in selected_periods:
                valid_mask = df[time_col].eq(period).to_numpy()
                train_mask = ~valid_mask
                train_idx = indices[train_mask]
                valid_idx = indices[valid_mask]
                if len(train_idx) and len(valid_idx):
                    lopo_splits.append((train_idx, valid_idx))
            add_design(
                "leave_one_period_out",
                lopo_splits,
                f"Leave one `{time_col}` period out, using the {recent_period_note} when many exist.",
                "time-aware diagnostic, but train folds may include periods after the validation period",
                adjustment_reason=f"`{time_col}` has {len(recent_periods)} unique periods",
            )

            time_k = min(requested_n_splits, len(periods))
            blocked_splits = _split_from_period_blocks(df, time_col, time_k)
            time_adjustment_reason = (
                f"`{time_col}` has {len(periods)} unique periods"
                if len(periods) < requested_n_splits
                else "forward-chaining needs an earlier training period before each validation block"
            )
            add_design(
                "time_blocked_split",
                blocked_splits,
                f"Forward-chaining blocked split ordered by `{time_col}`.",
                "safer for future-period generalization",
                adjustment_reason=time_adjustment_reason,
            )

            if group_col and group_col in df.columns:
                add_design(
                    "panel_time_split",
                    blocked_splits,
                    f"Panel-aware future-period split over `{time_col}` that respects time ordering in panel-like data with `{group_col}` entities.",
                    "safer for panel data with repeated entities over time",
                    adjustment_reason=time_adjustment_reason,
                )
            else:
                add_skip("panel_time_split", "requires both a time column and a group/entity column")
        else:
            reason = f"`{time_col}` has fewer than two non-missing periods"
            add_skip("leave_one_period_out", reason)
            add_skip("time_blocked_split", reason)
            add_skip("panel_time_split", reason)
    else:
        add_skip("leave_one_period_out", "no usable time column was detected")
        add_skip("time_blocked_split", "no usable time column was detected")
        add_skip("panel_time_split", "no usable time column was detected")

    results["_skipped"] = skipped
    return results


def _metric_name(task_type: str, metric: str | None) -> str:
    if metric:
        return metric.lower()
    return "accuracy" if task_type == "classification" else "mae"


def _score_fold(task_type: str, metric_name: str, y_true: pd.Series, y_pred: np.ndarray) -> float:
    if task_type == "classification":
        if metric_name == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric_name != "accuracy":
            raise ValueError(f"Unsupported classification metric: {metric_name!r}")
        return float(accuracy_score(y_true, y_pred))

    if metric_name != "mae":
        raise ValueError(f"Unsupported regression metric: {metric_name!r}")
    return float(mean_absolute_error(y_true, y_pred))


def _interpretation(split_name: str, lower_is_better: bool) -> str:
    if split_name == "random_kfold":
        return "IID diagnostic baseline; compare against safer designs for optimism."
    if split_name == "stratified_kfold":
        return "Target-balanced row-wise baseline; still unsafe for groups or time."
    if split_name == "group_kfold":
        return "Tests generalization to held-out groups/entities."
    if split_name == "group_shuffle_split":
        return "Repeated held-out group/entity diagnostic."
    if split_name == "leave_one_period_out":
        return "Tests sensitivity to individual held-out periods."
    if split_name == "time_blocked_split":
        return "Forward-looking estimate for future periods."
    if split_name == "panel_time_split":
        return "Forward-looking panel estimate across repeated entities."
    direction = "lower" if lower_is_better else "higher"
    return f"Compare {direction} scores under the same model and features."


def _optimism_warning(results: dict, task_type: str) -> dict | None:
    random_result = results.get("random_kfold")
    if not random_result:
        return None
    comparator_name = next((name for name in SAFE_SPLIT_PRIORITY if name in results), None)
    if comparator_name is None:
        return None

    random_mean = float(random_result["mean_score"])
    safe_result = results[comparator_name]
    safe_mean = float(safe_result["mean_score"])

    if task_type == "classification":
        diff = random_mean - safe_mean
        if diff <= 0.02:
            return None
        return {
            "detected": True,
            "comparator": comparator_name,
            "random_kfold_score": random_mean,
            "comparator_score": safe_mean,
            "absolute_difference": diff,
            "message": (
                f"Random KFold accuracy is {diff:.3f} higher than {comparator_name}. "
                "This may indicate optimistic validation caused by group or temporal leakage."
            ),
        }

    if not math.isfinite(safe_mean) or safe_mean <= 0 or random_mean >= safe_mean:
        return None
    optimism_pct = (safe_mean - random_mean) / safe_mean * 100.0
    if optimism_pct <= 5.0:
        return None
    return {
        "detected": True,
        "comparator": comparator_name,
        "random_kfold_score": random_mean,
        "comparator_score": safe_mean,
        "optimism_pct": optimism_pct,
        "message": (
            f"Random KFold likely overestimates generalization by {optimism_pct:.1f}% "
            f"relative to {comparator_name}."
        ),
    }


def evaluate_cv_designs(
    df: pd.DataFrame,
    target_col: str,
    plan: dict,
    task_type: str,
    metric: str | None = None,
    model_name: str = "hist_gradient_boosting",
    n_splits: int = 5,
    random_state: int = 42,
    max_rows: int = 50000,
) -> dict:
    """Train the same baseline model under candidate CV designs."""

    metric_name = _metric_name(task_type, metric)
    lower_is_better = metric_name == "mae"
    benchmark: dict[str, Any] = {
        "enabled": True,
        "target_col": target_col,
        "task_type": task_type,
        "model_name": model_name,
        "metric_name": metric_name,
        "lower_is_better": lower_is_better,
        "n_splits_requested": int(n_splits),
        "requested_n_splits": _requested_split_count(n_splits),
        "max_rows": int(max_rows),
        "n_rows_input": int(len(df)),
        "n_rows_evaluated": 0,
        "sampled": False,
        "results": {},
        "skipped": [],
        "optimism_warning": None,
        "feature_info": {},
    }

    if not target_col or target_col not in df.columns:
        benchmark["skipped"].append({"split": "cv_benchmark", "reason": "target column was not provided or not found"})
        return benchmark
    if max_rows <= 0:
        benchmark["skipped"].append({"split": "cv_benchmark", "reason": "max_rows must be positive"})
        return benchmark

    work_df = df.dropna(subset=[target_col]).copy()
    if work_df.empty:
        benchmark["skipped"].append({"split": "cv_benchmark", "reason": "all target values are missing"})
        return benchmark
    if len(work_df) > max_rows:
        work_df = work_df.sample(n=max_rows, random_state=random_state).sort_index()
        benchmark["sampled"] = True
    work_df = work_df.reset_index(drop=True)
    benchmark["n_rows_evaluated"] = int(len(work_df))

    exclude_cols = _names(plan.get("id_cols"))
    try:
        X, y, _preprocessor, feature_info = prepare_modeling_data(work_df, target_col, exclude_cols=exclude_cols)
        benchmark["feature_info"] = feature_info
    except Exception as exc:  # noqa: BLE001
        benchmark["skipped"].append({"split": "cv_benchmark", "reason": f"modeling data preparation failed: {exc}"})
        return benchmark

    sparse_ok = (model_name or "").lower() != "hist_gradient_boosting"
    preprocessor = _build_preprocessor(
        feature_info.get("numeric_features", []),
        feature_info.get("categorical_features", []),
        sparse_output=sparse_ok,
    )

    candidate_splits = generate_candidate_splits(work_df, plan, y=y, n_splits=n_splits, random_state=random_state)
    benchmark["skipped"].extend(candidate_splits.get("_skipped", []))

    base_model = _build_model_with_preprocessor(preprocessor, task_type, model_name, random_state)
    for split_name, split_info in candidate_splits.items():
        if split_name == "_skipped":
            continue

        fold_scores: list[float] = []
        fold_errors: list[str] = []
        for fold_idx, (train_idx, valid_idx) in enumerate(split_info["splits"], start=1):
            try:
                model = clone(base_model)
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                predictions = model.predict(X.iloc[valid_idx])
                score = _score_fold(task_type, metric_name, y.iloc[valid_idx], predictions)
                fold_scores.append(score)
            except Exception as exc:  # noqa: BLE001
                fold_errors.append(f"fold {fold_idx}: {exc}")

        if not fold_scores:
            reason = "all folds failed"
            if fold_errors:
                reason += "; " + " | ".join(fold_errors[:3])
            benchmark["skipped"].append({"split": split_name, "reason": reason})
            continue

        values = np.asarray(fold_scores, dtype=float)
        requested_for_split = int(split_info.get("requested_n_splits", _requested_split_count(n_splits)))
        candidate_actual_folds = int(split_info.get("actual_n_folds", len(split_info.get("splits", []))))
        actual_n_folds = int(len(values))
        adjustment_note = split_info.get("adjustment_note")
        if actual_n_folds != candidate_actual_folds:
            fold_failure_note = (
                f"Evaluated {actual_n_folds} of {candidate_actual_folds} constructed folds because some folds failed."
            )
            adjustment_note = f"{adjustment_note} {fold_failure_note}" if adjustment_note else fold_failure_note

        result = {
            "description": split_info.get("description", ""),
            "risk": split_info.get("risk", ""),
            "fold_scores": [float(value) for value in values],
            "mean_score": float(values.mean()),
            "std_score": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n_folds": actual_n_folds,
            "requested_n_splits": requested_for_split,
            "actual_n_folds": actual_n_folds,
            "model_name": model_name,
            "metric_name": metric_name,
            "lower_is_better": lower_is_better,
            "interpretation": _interpretation(split_name, lower_is_better),
        }
        if adjustment_note:
            result["adjustment_note"] = adjustment_note
        if fold_errors:
            result["fold_errors"] = fold_errors
        benchmark["results"][split_name] = result

    benchmark["optimism_warning"] = _optimism_warning(benchmark["results"], task_type)
    return benchmark
