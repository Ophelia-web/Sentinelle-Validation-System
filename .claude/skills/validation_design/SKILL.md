---
name: validation_design
description: Designs leakage-safe cross-validation strategies for tabular, grouped, temporal, and panel prediction tasks.
---

# Validation Design Skill

## When to use this skill

Use this skill when a tabular prediction task needs a statistically credible validation plan before modeling. It is especially useful when rows may share entities, groups, time periods, categories, locations, accounts, patients, stores, or other repeated units.

## Goals

- Inspect available data descriptions and CSV schemas.
- Identify candidate target, time, group/entity, id, and block/category columns.
- Classify the data as IID, grouped, temporal, panel, or uncertain.
- Recommend primary and secondary validation strategies.
- Warn against risky random KFold when group, time, or panel structure is detected.
- Optionally run a CV strategy comparison that evaluates random, grouped, temporal, and panel-aware validation designs under a fixed simple baseline model.
- Generate reproducible validation artifacts.

## Workflow

1. Read available data description files if present, such as README files, data dictionaries, competition descriptions, schema notes, or sample submission documentation.
2. Inspect CSV schemas and small representative samples.
3. Identify candidate target columns if the user has not provided one.
4. Run schema detection:
   - candidate time-like columns
   - candidate group/entity columns
   - candidate row identifier columns
   - candidate block/category columns
5. Infer the dataset structure:
   - IID
   - grouped
   - temporal
   - panel
   - uncertain
6. Assess validation risks.
7. Recommend a primary validation strategy and secondary alternatives.
8. If requested and a target column is available, run the CV strategy comparison:
   - compare random KFold, stratified KFold when feasible, grouped splits, temporal splits, and panel-aware splits
   - use only lightweight dependencies such as pandas, numpy, and scikit-learn
   - skip split designs that cannot be constructed safely and report reasons
   - treat comparison results as validation-design sensitivity evidence, not model optimization
9. Write required outputs:
   - `validation_design_report.md`
   - `validation_design_plan.json`
   - `split_code.py`
   - `cv_benchmark_results.csv` when the CV strategy comparison is requested

## Detection logic

- Time-like columns often have names containing date, time, timestamp, month, year, week, period, period_id, or report_period, or have datetime-like values.
- Group/entity-like columns often have names containing user, patient, customer, store, hospital, school, state, county, region, jurisdiction, site, entity, group, or id, and repeated values.
- ID-like columns often have high cardinality, are mostly unique, end with `_id`, or contain `id`.
- Block/category-like columns are low-cardinality categorical columns that are not row identifiers and not the target.
- Panel structure requires at least one repeated group/entity-like column and at least one time-like column with repeated observations across combinations.

## Recommendation logic

- IID regression: recommend `KFold` or `RepeatedKFold`.
- IID classification: recommend `StratifiedKFold`.
- Grouped data: recommend `GroupKFold` or `GroupShuffleSplit`.
- Temporal data: recommend `TimeSeriesSplit`, forward-chaining, or last-period holdout.
- Panel data: recommend blocked time split, leave-one-period-out, or rolling-origin split; avoid shuffled random KFold.
- Uncertain data: compare multiple split designs and report sensitivity.
- CV strategy comparison mode: train the same simple baseline under candidate split designs to detect whether random KFold appears optimistic relative to safer grouped, temporal, or panel-aware designs.

## Required outputs

- Markdown report summarizing detected task type, structure, candidate columns, risks, recommendations, designs to avoid, and limitations.
- JSON plan with machine-readable profile and recommendations.
- Python split code with reusable split functions and inferred column names inserted only as defaults.
- Optional comparison CSV and report section with fold metrics, skipped split reasons, and optimism warnings.

## Safety and robustness rules

- Do not hardcode a competition-specific schema as the only valid schema.
- Do not commit actual competition data or private data.
- Treat detection results as candidates, not proof.
- Prefer transparent heuristics over opaque automation.
- Ask users to confirm ambiguous target, time, group, and block columns when uncertainty is high.
- Preserve statistical validity over leaderboard convenience.
- Do not hardcode competition-specific columns or values in the CV strategy comparison.
- If no target column is provided, skip CV strategy comparison and still complete structural validation design.
