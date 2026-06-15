# [Award C] Sentinelle Validation Agent — leakage-safe cross-validation design for tabular data

> Team info
> | Legal name | Affiliation | Institutional email | Kaggle username |
> |---|---|---|---|
> | [Name 1] | [University / Company / Independent] | [email] | [kaggle_user] |
> | [Name 2] | [University / Company / Independent] | [email] | [kaggle_user] |
>
> Registered team name: [TeamName]
> GitHub repository: https://github.com/Ophelia-web/C

## What it does

Sentinelle Validation Agent is a reusable statistical module for designing leakage-safe cross-validation in tabular prediction tasks. It inspects a dataset schema, detects likely IID, grouped, temporal, or panel structure, flags validation risks, recommends safer validation strategies, and generates reproducible split code. The optional CV strategy comparison runs the same simple model under random, grouped, temporal, and panel-aware validation designs.

The module is intentionally small. Its default mode does not train models or perform AutoML. When requested, the CV strategy comparison trains the same simple baseline model under several split designs to reveal whether random KFold is overly optimistic.

## Demo link

Repository demo: `examples/validation_design_demo/README.md`

## Agent Design and Architecture

| Component | What it does |
|---|---|
| Core engine | Standalone Python validation-design module that profiles schema, detects validation risks, recommends safer cross-validation designs, and writes artifacts. |
| User interfaces | Local FastAPI + React web app and CLI. |
| Optional agent adapter | A Claude Code compatible skill file is included for agent-based workflows, but the core module runs without an LLM. |
| Brain / LLM | Optional. Any coding agent can follow the documented workflow; a Claude Code compatible skill file is provided for convenience. The core module runs without an LLM. |
| Planning | Decomposes validation design into schema detection, leakage checks, split recommendation, optional CV strategy comparison, and artifact generation. |
| Execution | Runs locally with Python; no external datasets or cloud APIs are required. |
| Observation | Generated reports, JSON plans, split code, and benchmark CSV files are inspected as artifacts. |

## Why this is useful for Kaggle and statistical analysis

Many Kaggle modeling mistakes happen because the validation split is too optimistic. Shuffled random KFold can leak information when:

- the same entity appears in training and validation
- rows have chronological order
- panel data repeats entities over time
- identifiers or high-cardinality categorical fields allow memorization
- performance differs across blocks or categories

Sentinelle helps participants document a validation design before leaderboard iteration. It encourages reproducible statistical reasoning and makes validation choices easier for teammates and reviewers to audit.

## How to run the demo

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

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

## Expected outputs

The command writes:

```text
artifacts/validation_design_demo/
├── validation_design_report.md
├── validation_design_plan.json
└── split_code.py
```

Comparison mode also writes:

```text
artifacts/validation_design_demo_v2/cv_benchmark_results.csv
```

For the synthetic panel demo, expected findings include:

- regression target
- panel-like structure
- repeated entity column
- time-like month column
- block/category column
- warning against shuffled random KFold
- recommendation for blocked time validation, leave-one-period-out, or rolling-origin style validation
- optional comparison evidence that random KFold can score better than safer group/time-aware splits
- `requested_n_splits` and `actual_n_folds` in comparison output, because `--n-splits` is treated as a maximum and some designs may use fewer folds when limited by rows, groups, periods, or class/bin counts

## Limitations

- The module uses heuristics and should be reviewed with domain knowledge.
- It cannot guarantee that a feature is available at prediction time.
- The optional CV strategy comparison trains simple baseline models only; it is validation-design sensitivity analysis and does not tune or select leaderboard models.
- Ambiguous schemas may require users to specify target, time, group, or block columns manually.
- Very small datasets may need custom validation plans rather than standard K-fold designs.
