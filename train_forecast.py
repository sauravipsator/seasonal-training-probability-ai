"""Forecast next-month food demand and return the item expected to trend most.

Input is transaction-level or daily/monthly item-level demand.  One row needs at
least date, food_item, and quantity_sold.  The remaining columns are optional
business drivers from the project brief.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

REQUIRED = {"date", "food_item", "quantity_sold"}
OPTIONAL_CATEGORICAL = [
    "city", "region", "festival", "weather", "local_event",
]
OPTIONAL_NUMERIC = [
    "public_holiday", "promotion_active", "vendor_availability",
]


def read_and_aggregate(paths: list[str]) -> pd.DataFrame:
    """Accept either the documented generic schema or the supplied sales exports."""
    missing_paths = [path for path in paths if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError("Input CSV not found: " + ", ".join(missing_paths))
    raw = pd.concat((pd.read_csv(path, low_memory=False) for path in paths), ignore_index=True)
    if not REQUIRED.issubset(raw.columns):
        export_fields = {"created_at", "item_name", "quantity"}
        if not export_fields.issubset(raw.columns):
            missing = REQUIRED - set(raw.columns)
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        raw = raw.rename(columns={
            "created_at": "date", "item_name": "food_item", "quantity": "quantity_sold",
            "station_name": "city", "railway_zone": "region", "food_type": "food_type",
        })
        # Cancelled/confirmed orders are not demand that was actually fulfilled.
        if "order_status" in raw:
            raw = raw[raw["order_status"].isin(["Delivered", "Partially Delivered"])]
        raw["promotion_active"] = (pd.to_numeric(raw.get("discount", 0), errors="coerce").fillna(0) > 0).astype(int)

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["quantity_sold"] = pd.to_numeric(raw["quantity_sold"], errors="coerce")
    raw = raw.dropna(subset=["date", "food_item", "quantity_sold"])
    raw["month"] = raw["date"].dt.to_period("M").dt.to_timestamp()

    present = [c for c in OPTIONAL_CATEGORICAL + OPTIONAL_NUMERIC + ["food_type", "is_veg"] if c in raw]
    aggregations = {"quantity_sold": "sum"}
    aggregations.update({c: "last" for c in present})
    named_aggregations = {column: (column, operation) for column, operation in aggregations.items()}
    if "outlet_name" in raw:
        named_aggregations["vendor_availability"] = ("outlet_name", "nunique")
    if "city" in raw:
        named_aggregations["city_coverage"] = ("city", "nunique")
    if "region" in raw:
        named_aggregations["region_coverage"] = ("region", "nunique")
    return raw.groupby(["month", "food_item"], as_index=False).agg(**named_aggregations)


def make_training_frame(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Do not manufacture zero-sales months: these CSVs cover selected months only.
    frame = monthly.sort_values(["food_item", "month"]).copy()
    group = frame.groupby("food_item", group_keys=False)["quantity_sold"]
    frame["lag_1_month"] = group.shift(1)
    frame["lag_3_month"] = group.shift(3)
    frame["rolling_3_month_mean"] = group.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    frame["target_month"] = frame.groupby("food_item")["month"].shift(-1)
    frame["next_month_quantity"] = group.shift(-1)
    # Only genuine consecutive month pairs are valid supervised training examples.
    training = frame[frame.target_month.eq(frame.month + pd.offsets.MonthBegin(1))].copy()
    training = training.dropna(subset=["next_month_quantity"])
    for dataset in (training,):
        dataset["month_number"] = dataset.target_month.dt.month
        dataset["year"] = dataset.target_month.dt.year
        dataset["quarter"] = dataset.target_month.dt.quarter
    latest_month = frame.month.max()
    future = frame[frame.month.eq(latest_month)].copy()
    forecast_month = latest_month + pd.offsets.MonthBegin(1)
    future["month_number"] = forecast_month.month
    future["year"] = forecast_month.year
    future["quarter"] = forecast_month.quarter
    return training, future


def build_pipeline(categorical: list[str], numeric: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("fill", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
            ("numeric", Pipeline([( "fill", SimpleImputer(strategy="median"))]), numeric),
        ],
        remainder="drop",
    )
    model = XGBRegressor(
        objective="reg:squarederror", n_estimators=350, max_depth=4,
        learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
        random_state=42, n_jobs=1,
    )
    return Pipeline([( "prepare", preprocessor), ("model", model)])


def run(csv_path: str, output_path: str) -> pd.DataFrame:
    monthly = read_and_aggregate(csv_path)
    training, future = make_training_frame(monthly)
    if len(training) < 8:
        raise ValueError("Need at least 8 usable item-month observations; add more monthly history.")

    categorical = ["food_item"] + [c for c in OPTIONAL_CATEGORICAL if c in training]
    numeric = ["month_number", "year", "quarter", "lag_1_month", "lag_3_month", "rolling_3_month_mean"]
    numeric += [c for c in OPTIONAL_NUMERIC + ["city_coverage", "region_coverage"] if c in training]
    features = categorical + numeric

    # Chronological holdout: never train on a month after the month being tested.
    cutoff = training.target_month.max()
    train_rows, test_rows = training[training.target_month < cutoff], training[training.target_month == cutoff]
    if len(train_rows) >= 4 and len(test_rows):
        validation_model = build_pipeline(categorical, numeric)
        validation_model.fit(train_rows[features], train_rows.next_month_quantity)
        predictions = validation_model.predict(test_rows[features]).clip(0)
        mape = mean_absolute_percentage_error(test_rows.next_month_quantity, predictions) * 100
        print(f"Chronological holdout MAPE: {mape:.2f}%")

    model = build_pipeline(categorical, numeric)
    model.fit(training[features], training.next_month_quantity)
    future["forecast_quantity"] = model.predict(future[features]).clip(0)
    forecast_month = future.month.max() + pd.offsets.MonthBegin(1)
    result = future[["food_item", "forecast_quantity"]].sort_values("forecast_quantity", ascending=False)
    result.insert(0, "forecast_month", forecast_month.strftime("%Y-%m"))
    result.insert(1, "rank", range(1, len(result) + 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    winner = result.iloc[0]
    print(f"Trending food for {winner.forecast_month}: {winner.food_item} "
          f"(forecast demand: {winner.forecast_quantity:.0f})")
    print(f"Saved ranked forecast: {output_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="One or more source demand CSVs")
    parser.add_argument("--output", default="output/monthly_food_forecast.csv")
    args = parser.parse_args()
    run(args.input, args.output)
