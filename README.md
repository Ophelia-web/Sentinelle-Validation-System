# Sentinelle Validation Agent

<p align="center">
  <img src="assets/C_logo.png" alt="Sentinelle Validation System logo" width="220">
</p>

**Leakage-safe cross-validation design for tabular data.**

Sentinelle Validation Agent is a small reusable statistical agent/module that inspects a tabular prediction dataset, detects likely IID, grouped, temporal, or panel structure, and produces a reproducible validation-risk report plus split code. The optional CV strategy comparison runs the same simple model under random, grouped, temporal, and panel-aware validation designs to show whether random splits may be overly optimistic.

## Why this matters

Most modeling mistakes begin before the model is trained: the validation split is wrong. Random KFold can look reassuring while leaking information across repeated entities, future periods, duplicated identifiers, or category blocks. Sentinelle focuses on the statistical validity of model evaluation, not on AutoML or model selection.

Use it when you want a quick, transparent answer to:

- Does my dataset look IID, grouped, temporal, panel-like, or uncertain?
- Which columns look like time, group/entity, row ID, target, or category/block columns?
- Which validation designs are safer than shuffled random KFold?
- Does random KFold look overly optimistic compared with grouped or time-aware validation?
- What split code should I start from?

## 🎬 Demo Video

<p align="center">
  <a href="https://www.youtube.com/watch?v=Ggc8ZP8gDks">
    <img 
      src="https://img.youtube.com/vi/Ggc8ZP8gDks/maxresdefault.jpg"
      width="90%"
      style="border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.15);"
    />
  </a>
</p>

Click the image above to watch the full demo on YouTube.
👉 https://www.youtube.com/watch?v=Ggc8ZP8gDks

This demo shows:
- Uploading a dataset
- Detecting panel / time / group structure
- Comparing CV strategies
- Interpreting leakage risks
- Exporting validation artifacts

## Quickstart

Install minimal dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the synthetic panel demo in fast structural-diagnosis mode:

```bash
python3 examples/validation_design_demo/make_mock_panel_data.py
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

Expected artifacts:

```text
artifacts/validation_design_demo/
├── validation_design_report.md
├── validation_design_plan.json
└── split_code.py
```

Comparison mode also writes:

```text
artifacts/validation_design_demo_v2_n3/cv_benchmark_results.csv
```

## Local Web Interface

Sentinelle includes a local web interface for interactive validation diagnostics.

The web app runs on your own machine. It does not require a hosted service,
external API, or cloud deployment. Users can upload a CSV training table, select
a target column, detect time/group/panel structure, compare validation
strategies, and download generated artifacts including a PDF report, JSON plan,
reusable split code, and optional benchmark results.

### Run locally

Install Python dependencies:

```bash
python -m pip install -r requirements-web.txt
```

Build the React frontend:

```bash
cd frontend
npm install
npm run build
cd ..
```

Start the local FastAPI server:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Recommended workflow:

1. Click **Generate sample dataset**.
2. Keep `target` as the target column.
3. Enable **Compare CV strategies**.
4. Use `random_forest` and `n_splits = 3`.
5. Click **Analyze dataset**.
6. Review the detected panel structure, high leakage risk, blocked-time
   validation recommendation, comparison chart, PDF report, and downloadable
   artifacts.

### Optional frontend development mode

Terminal 1:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Example output

For the mock panel dataset, Sentinelle detects repeated entities across time and recommends panel-aware validation:

```text
Detected structure: panel
Primary CV: blocked_time_split
Risk level: high
Avoid: shuffled random KFold because repeated groups and time-like structure are present.
```

The generated `split_code.py` includes reusable functions such as:

- `make_kfold_splits`
- `make_group_kfold_splits`
- `make_leave_one_period_out_splits`
- `make_time_blocked_splits`

When `--run-cv-benchmark` is supplied with a target column, Sentinelle runs a CV strategy comparison: the same lightweight baseline model is evaluated under candidate split designs and reports metrics such as random KFold MAE versus group- or time-aware MAE. The comparison is diagnostic: it is designed to reveal validation sensitivity, not to tune or optimize a leaderboard model. `--n-splits` is treated as a maximum requested split count; individual designs may use fewer folds when limited by the number of rows, unique groups, unique time periods, or class/bin counts. Outputs include both `requested_n_splits` and `actual_n_folds` so fold-count downgrades are visible.

## Repository structure

```text
.
├── app.py                          # FastAPI backend + production static frontend serving
├── requirements-web.txt            # backend/runtime Python dependencies
├── frontend/                       # React + Vite web app
│   ├── package.json
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── styles.css
├── .claude/skills/validation_design/SKILL.md
├── docs/
│   ├── award_c_validation_agent.md
│   └── kaggle_discussion_draft.md
├── examples/validation_design_demo/
│   ├── README.md
│   ├── make_mock_panel_data.py
│   └── expected_output.md
├── scripts/validation_design_agent.py
└── src/validation_design/
    ├── cv_benchmark.py
    ├── schema_detector.py
    ├── split_recommender.py
    ├── leakage_checks.py
    ├── split_templates.py
    └── report.py
```

## Optional Agent Skill Adapter

Sentinelle runs as a standalone Python CLI and local web application. The repository also includes an optional Claude Code compatible skill file at `.claude/skills/validation_design/SKILL.md`.

This skill file describes how an agent can apply the same validation-design workflow inside a project. The core validation engine, CLI, and local web interface do not require Claude, external APIs, or proprietary cloud services.

## Limitations

- Sentinelle uses heuristics; it cannot replace domain knowledge.
- The optional CV strategy comparison trains simple baseline models only; it does not tune models or prove that one validation design is optimal.
- Column-name inference can be ambiguous in poorly documented datasets.
- Small datasets, rare blocks, and sparse groups may require manual adjustment.
- It does not use private competition data and should not be treated as dataset-specific.

## Award C positioning

Sentinelle Validation Agent is designed for Award C: Statistical Skill / Agent Module in the STAI-X Challenge 2026. It is intentionally general-purpose for tabular prediction tasks while demonstrating why public-health-style panel data should be evaluated with group/time-aware validation rather than naive random splits.
