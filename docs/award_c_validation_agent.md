# Sentinelle Validation Agent for Award C

## What it does

Sentinelle Validation Agent is a reusable statistical module for leakage-safe cross-validation design in tabular prediction tasks. It profiles a dataset, detects likely IID, grouped, temporal, or panel structure, flags validation risks, recommends safer split strategies, and writes reproducible artifacts. The optional CV strategy comparison evaluates random, grouped, temporal, and panel-aware validation designs under the same simple model.

It produces:

- `validation_design_report.md`
- `validation_design_plan.json`
- `split_code.py`
- `cv_benchmark_results.csv` when CV strategy comparison is requested

## Why it matters statistically

Validation is a statistical design problem. If validation rows are not independent from training rows, measured performance can be biased. Common examples include:

- the same patient, customer, store, school, region, or state appearing in both train and validation folds
- future periods leaking into training when validation is meant to approximate future prediction
- row identifiers or high-cardinality categorical fields enabling memorization
- category blocks with uneven performance being hidden by an aggregate metric

Sentinelle does not search for the best model. It helps users protect the validity of the evaluation protocol before modeling begins.

## Who can use it

The module is designed for Kaggle participants, applied statisticians, public health modelers, data scientists, and agent builders working with tabular datasets. It is general-purpose and supports arbitrary column names through heuristics rather than fixed schemas.

## How it works

1. Profile the dataframe shape, dtypes, missingness, cardinality, and candidate target columns.
2. Detect time-like columns using names and datetime-like values.
3. Detect group/entity-like columns using names and repeated values.
4. Detect row identifier columns using high uniqueness and id-like names.
5. Detect low-cardinality block/category columns for subgroup reporting.
6. Infer the data structure:
   - IID
   - grouped
   - temporal
   - panel
   - uncertain
7. Assess leakage and robustness risks.
8. Recommend primary and secondary validation designs.
9. Optionally train the same simple baseline model under multiple validation designs and compare CV metrics.
10. Generate reusable split code.

## Demo walkthrough

Generate synthetic panel data:

```bash
python3 examples/validation_design_demo/make_mock_panel_data.py
```

Run the agent in fast structural-diagnosis mode:

```bash
python3 scripts/validation_design_agent.py \
  --train examples/validation_design_demo/mock_panel_data.csv \
  --target target \
  --output-dir artifacts/validation_design_demo
```

Run the optional CV strategy comparison with up to 3 folds per design:

```bash
python3 scripts/validation_design_agent.py \
  --train examples/validation_design_demo/mock_panel_data.csv \
  --target target \
  --output-dir artifacts/validation_design_demo_v2_n3 \
  --run-cv-benchmark \
  --model-name random_forest \
  --n-splits 3
```

Run the same comparison with up to 5 folds per design:

```bash
python3 scripts/validation_design_agent.py \
  --train examples/validation_design_demo/mock_panel_data.csv \
  --target target \
  --output-dir artifacts/validation_design_demo_v2_n5 \
  --run-cv-benchmark \
  --model-name random_forest \
  --n-splits 5
```

Use `python3` if `python` is not available. If your environment maps Python 3 to `python`, the same commands also work with `python`.

The demo data contains repeated states over months and outcome categories. Sentinelle should detect a panel-like structure, warn against shuffled random KFold, and recommend blocked time validation with alternatives such as leave-one-period-out. In comparison mode, the agent trains the same simple baseline under random, grouped, temporal, and panel-aware splits to show whether random KFold gives an overly optimistic metric. This is sensitivity analysis for validation design, not tuning or selecting a leaderboard model. `--n-splits` is a maximum requested split count, so designs can use fewer actual folds when rows, groups, periods, or class/bin counts are limiting.

## Files produced

- `validation_design_report.md`: human-readable explanation for modelers and reviewers.
- `validation_design_plan.json`: machine-readable profile, risks, and recommendations.
- `split_code.py`: reusable Python functions for KFold, GroupKFold, leave-one-period-out, and blocked time splits.
- `cv_benchmark_results.csv`: optional CV strategy comparison metrics by validation design.

## Optional Agent Skill Adapter

The repository includes `.claude/skills/validation_design/SKILL.md`, which describes how an agent can apply this workflow inside a project. This Claude Code compatible skill file is optional; the core validation module, CLI, and local web interface run without an LLM.

## Generality beyond STAI-X

Although the STAI-X Challenge motivates this module, Sentinelle is not hardcoded to the overdose dataset. It does not assume specific target, period, jurisdiction, or category names. Instead, it uses column-name hints and data profiling to infer candidate time, group, id, and block columns.

This makes it applicable to many tabular prediction settings, including:

- public health panels
- retail store forecasting
- patient-level clinical prediction
- education or school-level modeling
- customer/account churn
- regional economic indicators

## Limitations

- Heuristics can be wrong or incomplete.
- Domain knowledge is required to confirm prediction-time feature availability.
- The default mode does not train models; the optional CV strategy comparison uses simple baseline models and is diagnostic rather than leaderboard-oriented.
- The tool does not prove that the recommended validation design is optimal.
- Small group counts and short time series can require custom holdouts.
- Generated split code should be reviewed before production use.
