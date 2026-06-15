# Expected Output Excerpt

After running:

```bash
python3 examples/validation_design_demo/make_mock_panel_data.py
python3 scripts/validation_design_agent.py \
  --train examples/validation_design_demo/mock_panel_data.csv \
  --target target \
  --output-dir artifacts/validation_design_demo
```

The CLI should print a concise summary similar to:

```text
Sentinelle Validation Agent
Source: examples/validation_design_demo/mock_panel_data.csv
Rows profiled: 648
Detected task type: regression
Detected structure: panel
Primary CV: blocked_time_split
Risk level: high
Artifacts written to: artifacts/validation_design_demo
```

The generated `validation_design_report.md` should include:

```markdown
## Summary

- Detected task type: **regression**
- Detected data structure: **panel**
- Risk level: **high**
- Primary validation design: **blocked_time_split**

## Leakage risks

| Severity | Risk | Evidence | Recommendation |
|---|---|---|---|
| high | Random KFold may leak repeated groups | Column `state` has repeated values. | Use GroupKFold, GroupShuffleSplit, or a group-aware holdout. |
| high | Random KFold may leak future information | Time-like columns detected: `month`. | Use forward-chaining, last-period holdout, rolling-origin, or blocked time splits. |
```

Exact scores and table ordering may vary as heuristics evolve.

After running the optional CV strategy comparison:

```bash
python3 scripts/validation_design_agent.py \
  --train examples/validation_design_demo/mock_panel_data.csv \
  --target target \
  --output-dir artifacts/validation_design_demo_v2_n5 \
  --run-cv-benchmark \
  --model-name random_forest \
  --n-splits 5
```

The output directory should also include:

```text
cv_benchmark_results.csv
```

The report should include an `## CV Strategy Comparison` section with random, grouped, temporal, or panel-aware metrics when those splits can be constructed. Exact metric values are stochastic, but random KFold should be treated as a diagnostic baseline and compared against safer splits.

`--n-splits` is interpreted as a maximum requested split count. The CSV and report include `requested_n_splits` and `actual_n_folds` so users can see when a design uses fewer folds because rows, groups, periods, or class/bin counts are limiting. The comparison is validation-design sensitivity analysis, not leaderboard model tuning.
