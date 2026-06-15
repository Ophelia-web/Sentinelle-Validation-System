"""Sentinelle Validation Agent package."""

from validation_design.cv_benchmark import (
    build_baseline_model,
    evaluate_cv_designs,
    generate_candidate_splits,
    prepare_modeling_data,
)
from validation_design.leakage_checks import assess_validation_risks
from validation_design.report import write_json_plan, write_markdown_report
from validation_design.schema_detector import (
    detect_block_like_columns,
    detect_group_like_columns,
    detect_id_like_columns,
    detect_time_like_columns,
    infer_data_structure,
    infer_task_type,
    profile_dataframe,
)
from validation_design.split_recommender import recommend_validation_strategy
from validation_design.split_templates import generate_split_code

__all__ = [
    "assess_validation_risks",
    "build_baseline_model",
    "detect_block_like_columns",
    "detect_group_like_columns",
    "detect_id_like_columns",
    "detect_time_like_columns",
    "evaluate_cv_designs",
    "generate_candidate_splits",
    "generate_split_code",
    "infer_data_structure",
    "infer_task_type",
    "prepare_modeling_data",
    "profile_dataframe",
    "recommend_validation_strategy",
    "write_json_plan",
    "write_markdown_report",
]
