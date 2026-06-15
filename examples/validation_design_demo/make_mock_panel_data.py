"""Create a small synthetic panel dataset for Sentinelle demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_mock_panel_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    states = ["AL", "AZ", "CA", "CO", "FL", "GA", "IL", "MA", "NY", "OH", "TX", "WA"]
    months = pd.period_range("2023-01", periods=18, freq="M").astype(str)
    outcome_types = ["primary", "secondary", "combined"]
    area_blocks = {
        "AL": "south",
        "AZ": "west",
        "CA": "west",
        "CO": "west",
        "FL": "south",
        "GA": "south",
        "IL": "midwest",
        "MA": "northeast",
        "NY": "northeast",
        "OH": "midwest",
        "TX": "south",
        "WA": "west",
    }

    rows = []
    state_effects = {state: rng.normal(0.0, 1.2) for state in states}
    state_signal_a = {state: rng.normal(0.0, 1.0) for state in states}
    state_signal_b = {state: rng.normal(0.0, 1.0) for state in states}
    area_block_effects = {"south": 0.35, "west": -0.25, "midwest": 0.05, "northeast": -0.1}
    outcome_effects = {"primary": -0.45, "secondary": 0.2, "combined": 0.65}
    for month_idx, month in enumerate(months):
        seasonal = np.sin(month_idx / 12 * 2 * np.pi)
        trend = 0.18 * month_idx + 0.025 * max(0, month_idx - 11) ** 2
        for state in states:
            area_block = area_blocks[state]
            for outcome_type in outcome_types:
                population_scaled = state_signal_a[state] + rng.normal(0, 0.12)
                service_index = 0.7 + 0.35 * state_signal_b[state] + 0.03 * month_idx + rng.normal(0, 0.08)
                reporting_lag = (month_idx % 4) + rng.normal(0, 0.15)
                target = (
                    5.0
                    + state_effects[state]
                    + area_block_effects[area_block]
                    + outcome_effects[outcome_type]
                    + trend
                    + 0.5 * seasonal
                    + 0.55 * population_scaled
                    - 0.25 * service_index
                    + 0.12 * reporting_lag
                    + rng.normal(0, 0.18)
                )
                rows.append(
                    {
                        "row_id": f"{state}-{month}-{outcome_type}",
                        "state": state,
                        "month": month,
                        "area_block": area_block,
                        "outcome_type": outcome_type,
                        "population_scaled": population_scaled,
                        "service_index": service_index,
                        "reporting_lag": reporting_lag,
                        "target": target,
                    }
                )
    return pd.DataFrame(rows)


def main() -> int:
    output_path = Path(__file__).with_name("mock_panel_data.csv")
    df = make_mock_panel_data()
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df):,} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
