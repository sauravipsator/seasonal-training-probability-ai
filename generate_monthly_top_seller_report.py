"""Build the observed-month top-seller report from the supplied sales exports."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

INPUTS = sorted(Path("data").glob("[0-9]*.csv"))
OUTPUT = Path("output/monthly_top_sellers.csv")
SUMMARY = Path("tmp/monthly_report_summary.json")


def main() -> None:
    if not INPUTS:
        raise FileNotFoundError("No cleaned_*.csv source files found in data/")

    raw = pd.concat((pd.read_csv(path, low_memory=False) for path in INPUTS), ignore_index=True)
    raw = raw[raw["order_status"].isin(["Delivered", "Partially Delivered"])].copy()
    raw["month"] = pd.to_datetime(raw["created_at"], errors="coerce").dt.to_period("M").astype(str)
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce").fillna(0)
    raw = raw.dropna(subset=["month", "item_name"])

    monthly = (
        raw.groupby(["month", "item_name"], as_index=False)
        .agg(
            units_sold=("quantity", "sum"),
            order_lines=("order_id", "size"),
            active_vendors=("outlet_name", "nunique"),
            active_stations=("station_name", "nunique"),
        )
        .sort_values(["month", "units_sold", "order_lines"], ascending=[True, False, False])
    )
    monthly["rank"] = monthly.groupby("month").cumcount() + 1
    winners = monthly[monthly["rank"] == 1].copy()
    winners = winners[["month", "rank", "item_name", "units_sold", "order_lines", "active_vendors", "active_stations"]]
    winners = winners.rename(columns={"item_name": "top_seller"})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    winners.to_csv(OUTPUT, index=False)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps({
        "rows": winners.to_dict(orient="records"),
        "source_files": [path.name for path in INPUTS],
        "observed_months": int(len(winners)),
        "total_delivered_units": int(raw.quantity.sum()),
    }))
    print(winners.to_string(index=False))


if __name__ == "__main__":
    main()
