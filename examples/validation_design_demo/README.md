# Validation Design Demo

This demo creates a synthetic panel dataset with repeated entities across time. It does not use competition data. The data includes persistent entity effects, temporal trend, block/category effects, and entity-pattern features so random row-wise CV is likely to look better than safer group/time-aware validation.

## Generate mock data

```bash
python3 examples/validation_design_demo/make_mock_panel_data.py
```

This writes:

```text
examples/validation_design_demo/mock_panel_data.csv
```

The CSV is intentionally ignored by git because generated data artifacts should not be committed.

## Run the agent: fast structural diagnosis

```bash
python3 scripts/validation_design_agent.py \
  --train examples/validation_design_demo/mock_panel_data.csv \
  --target target \
  --output-dir artifacts/validation_design_demo
```

## Run the optional CV strategy comparison

```bash
python3 scripts/validation_design_agent.py \
  --train examples/validation_design_demo/mock_panel_data.csv \
  --target target \
  --output-dir artifacts/validation_design_demo_v2_n3 \
  --run-cv-benchmark \
  --model-name random_forest \
  --n-splits 3
```

```bash
python3 scripts/validation_design_agent.py \
  --train examples/validation_design_demo/mock_panel_data.csv \
  --target target \
  --output-dir artifacts/validation_design_demo_v2_n5 \
  --run-cv-benchmark \
  --model-name random_forest \
  --n-splits 5
```

The CV strategy comparison trains the same simple baseline model under different validation designs to reveal whether random KFold gives overly optimistic scores. It is sensitivity analysis, not leaderboard tuning. `--n-splits` is a maximum requested split count; designs may use fewer actual folds when the data has fewer valid rows, groups, periods, or class/bin members.

Use `python3` if `python` is not available. If your environment maps Python 3 to `python`, the same commands also work with `python`.

## Expected outputs

```text
artifacts/validation_design_demo/
├── validation_design_report.md
├── validation_design_plan.json
└── split_code.py
```

Comparison mode additionally writes:

```text
artifacts/validation_design_demo_v2_n3/cv_benchmark_results.csv
```

The report should identify:

- `month` as a time-like column
- `state` as a group/entity-like column
- `row_id` as an identifier-like column
- `outcome_type` as a block/category column
- panel-like structure because states repeat across months
- high risk for random KFold when repeated entities and future periods are mixed across folds

## Why this illustrates the problem

A shuffled random KFold split can place the same state and neighboring months in both training and validation folds. That can make validation error look better than a future-facing deployment setting. A panel-aware split, such as blocked time validation or leave-one-period-out validation, better tests whether a model generalizes across future periods and repeated entities.
