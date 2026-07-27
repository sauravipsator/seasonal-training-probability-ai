"""Create the JSON read by the Java dashboard API from the production CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_FILES = sorted(Path("data").glob("[0-9]*.csv"))
OUT = Path("output/dashboard-data.json")
STATE_MAP = Path("data/station_state_map.csv")
CANCELLATION_METRICS = Path("output/cancellation_model_metrics.json")


def clean_text(series: pd.Series, fallback: str) -> pd.Series:
    return series.fillna(fallback).astype(str).str.strip().replace("", fallback)


def main() -> None:
    raw = pd.concat((pd.read_csv(path, low_memory=False) for path in DATA_FILES), ignore_index=True)
    raw["booking_date"] = pd.to_datetime(raw["booking_date"], errors="coerce")
    raw = raw.dropna(subset=["booking_date", "order_id"]).copy()
    raw["month"] = raw["booking_date"].dt.to_period("M").astype(str)
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce").fillna(0)
    raw["gross_item_amount"] = pd.to_numeric(raw["gross_item_amount"], errors="coerce").fillna(0)
    raw["station"] = clean_text(raw["station_name"], "Unknown station")
    raw["station_code"] = clean_text(raw["station_code"], "Unknown")
    raw["item"] = clean_text(raw["item_name"], "Unknown item")
    raw["status"] = clean_text(raw["order_status"], "Unknown")
    raw["is_cancelled"] = raw.status.str.contains("cancel", case=False, na=False)
    raw["is_fulfilled"] = raw.status.isin(["Delivered", "Partially Delivered"])

    # The source data has station codes but no state. Create a user-editable map
    # once; state holidays can be joined after its `state` column is populated.
    stations = raw[["station_code", "station"]].drop_duplicates().sort_values(["station_code", "station"])
    if STATE_MAP.exists():
        state_map = pd.read_csv(STATE_MAP, dtype=str).fillna("")
    else:
        state_map = stations.assign(state="")
        state_map.to_csv(STATE_MAP, index=False)
    raw = raw.merge(state_map[["station_code", "state"]].drop_duplicates("station_code"), on="station_code", how="left")
    state_coverage = float(raw.state.fillna("").ne("").mean())

    fulfilled = raw[raw.is_fulfilled].copy()
    item_month = (
        fulfilled.groupby(["month", "station", "item"], as_index=False)
        .agg(units=("quantity", "sum"), order_lines=("order_id", "size"), vendors=("outlet_name", "nunique"))
        .sort_values(["month", "station", "units", "order_lines"], ascending=[True, True, False, False])
    )
    item_month["rank"] = item_month.groupby(["month", "station"]).cumcount() + 1
    top_items = item_month[item_month["rank"] <= 10].copy()

    # `amount_payable` is repeated on individual item rows, so revenue is calculated
    # once per order; gross_item_amount is used only as a safe fallback.
    raw["order_revenue"] = pd.to_numeric(raw["amount_payable"], errors="coerce")
    orders = (
        raw.sort_values("booking_date").groupby("order_id", as_index=False)
        .agg(month=("month", "first"), status=("status", "last"), is_cancelled=("is_cancelled", "max"),
             revenue=("order_revenue", "max"), fallback_revenue=("gross_item_amount", "sum"))
    )
    orders["revenue"] = orders.revenue.fillna(orders.fallback_revenue).fillna(0)
    monthly_orders = (
        orders.groupby("month", as_index=False)
        .agg(order_count=("order_id", "size"), cancellation_count=("is_cancelled", "sum"),
             revenue=("revenue", "sum"), revenue_loss=("revenue", lambda s: s[orders.loc[s.index, "is_cancelled"]].sum()))
    )
    monthly_orders["cancellation_rate"] = monthly_orders.cancellation_count / monthly_orders.order_count

    global_items = (
        fulfilled.groupby("item", as_index=False)
        .agg(units=("quantity", "sum"), observed_months=("month", "nunique"), latest_month=("month", "max"))
        .sort_values("units", ascending=False)
    )
    forecast_path = Path("output/monthly_food_forecast.csv")
    forecast = pd.read_csv(forecast_path) if forecast_path.exists() else pd.DataFrame()
    latest = fulfilled.month.max()
    latest_items = fulfilled[fulfilled.month.eq(latest)].groupby("item", as_index=False).quantity.sum().rename(columns={"quantity": "latest_units"})
    if not forecast.empty:
        forecast = forecast.rename(columns={"food_item": "item", "forecast_quantity": "forecast_units"})
        forecast = forecast.merge(global_items[["item", "units", "observed_months"]], on="item", how="left")
        forecast = forecast.merge(latest_items, on="item", how="left").head(10)
        forecast["reason"] = forecast.apply(
            lambda r: f"{int(r.observed_months or 0)} observed months; {int(r.units or 0):,} historical units; {int(r.latest_units or 0):,} units in {latest}.", axis=1
        )

    payload = {
        "metadata": {
            "model": "XGBoost demand trend prototype",
            "forecast_month": str(forecast.iloc[0].forecast_month) if not forecast.empty else None,
            "latest_observed_month": latest,
            "data_quality_note": "Only selected production months were supplied. Forecast confidence is directional, not production-grade.",
            "cancellation_model_available": bool(raw.is_cancelled.any()),
            "cancellation_note": (
                "Cancelled orders are available. A cancellation-risk model can now be trained using booking time, ETA lead time, station, vendor, item, price, and discount."
                if raw.is_cancelled.any() else "No cancelled orders were found in the supplied data, so cancellation-risk training is unavailable."
            ),
            "state_mapping_coverage": state_coverage,
            "state_mapping_note": "Fill data/station_state_map.csv to enable station-specific state-holiday features for 2026.",
        },
        "monthly": monthly_orders.sort_values("month").to_dict(orient="records"),
        "topItems": top_items.to_dict(orient="records"),
        "forecast": forecast[["forecast_month", "rank", "item", "forecast_units", "units", "observed_months", "latest_units", "reason"]].to_dict(orient="records") if not forecast.empty else [],
        "cancellationModel": json.loads(CANCELLATION_METRICS.read_text()) if CANCELLATION_METRICS.exists() else None,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, default=lambda value: int(value) if hasattr(value, "item") else str(value)))
    print(f"Wrote {OUT} with {len(top_items):,} station-item rankings.")


if __name__ == "__main__":
    main()
