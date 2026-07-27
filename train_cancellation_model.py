"""Train an XGBoost cancellation-risk model using booking-time information only."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

INPUTS = sorted(Path("data").glob("[0-9]*.csv"))
OUTPUT = Path("output/cancellation_model_metrics.json")
MODEL_OUTPUT = Path("output/cancellation_model.joblib")
MAX_TRAIN_ROWS = 300_000


def make_order_frame() -> pd.DataFrame:
    raw = pd.concat((pd.read_csv(path, low_memory=False) for path in INPUTS), ignore_index=True)
    raw["booking_date"] = pd.to_datetime(raw["booking_date"], errors="coerce")
    raw["eta"] = pd.to_datetime(raw["eta"], errors="coerce")
    for column in ["quantity", "selling_price", "discount"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(0)
    raw = raw.dropna(subset=["booking_date", "order_id"]).sort_values("booking_date")
    raw["cancelled"] = raw["order_status"].fillna("").str.contains("cancel", case=False)
    orders = raw.groupby("order_id", as_index=False).agg(
        booking_date=("booking_date", "first"), eta=("eta", "first"), cancelled=("cancelled", "max"),
        station_code=("station_code", "first"), outlet_name=("outlet_name", "first"),
        food_type=("food_type", "first"), payment_type=("payment_type", "first"), is_veg=("is_veg", "first"),
        item_count=("item_name", "nunique"), total_quantity=("quantity", "sum"),
        total_selling_price=("selling_price", "sum"), total_discount=("discount", "sum"),
    )
    orders["booking_month"] = orders.booking_date.dt.month
    orders["booking_day_of_week"] = orders.booking_date.dt.dayofweek
    orders["booking_hour"] = orders.booking_date.dt.hour
    orders["is_weekend"] = orders.booking_day_of_week.ge(5).astype(int)
    orders["eta_hour"] = orders.eta.dt.hour
    orders["eta_lead_hours"] = (orders.eta - orders.booking_date).dt.total_seconds() / 3600
    return orders


def main() -> None:
    orders = make_order_frame()
    test_month = orders.booking_date.dt.to_period("M").max()
    train = orders[orders.booking_date.dt.to_period("M") < test_month].copy()
    test = orders[orders.booking_date.dt.to_period("M") == test_month].copy()
    if len(train) > MAX_TRAIN_ROWS:
        sample_size = min(MAX_TRAIN_ROWS // 2, int(train.cancelled.value_counts().min()))
        train = train.groupby("cancelled", group_keys=False).sample(n=sample_size, random_state=42)

    categorical = ["station_code", "outlet_name", "food_type", "payment_type", "is_veg"]
    numeric = ["booking_month", "booking_day_of_week", "booking_hour", "is_weekend", "eta_hour", "eta_lead_hours",
               "item_count", "total_quantity", "total_selling_price", "total_discount"]
    processor = ColumnTransformer([
        ("category", Pipeline([( "fill", SimpleImputer(strategy="most_frequent")),
                                ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20))]), categorical),
        ("number", SimpleImputer(strategy="median"), numeric),
    ])
    model = XGBClassifier(objective="binary:logistic", eval_metric="auc", n_estimators=240, max_depth=5,
        learning_rate=0.06, subsample=0.8, colsample_bytree=0.8, min_child_weight=8, random_state=42, n_jobs=4)
    pipeline = Pipeline([( "prepare", processor), ("model", model)])
    pipeline.fit(train[categorical + numeric], train.cancelled.astype(int))
    probability = pipeline.predict_proba(test[categorical + numeric])[:, 1]
    names = pipeline.named_steps["prepare"].get_feature_names_out()
    def original_feature(name: str) -> str:
        if name.startswith("number__"):
            return name.removeprefix("number__")
        value = name.removeprefix("category__")
        return next((column for column in categorical if value == column or value.startswith(f"{column}_")), value)
    factors = (pd.DataFrame({"feature": names, "importance": pipeline.named_steps["model"].feature_importances_})
        .assign(feature=lambda x: x.feature.map(original_feature))
        .groupby("feature", as_index=False).importance.sum().sort_values("importance", ascending=False).head(8))
    result = {"status": "trained", "test_month": str(test_month), "training_orders": int(len(train)),
        "test_orders": int(len(test)), "test_cancellation_rate": float(test.cancelled.mean()),
        "roc_auc": float(roc_auc_score(test.cancelled, probability)),
        "average_precision": float(average_precision_score(test.cancelled, probability)),
        "factors": factors.to_dict(orient="records"),
        "note": "Risk uses information available at booking time only; use it to prioritise review, not automatically reject orders."}
    joblib.dump(pipeline, MODEL_OUTPUT)
    OUTPUT.write_text(json.dumps(result))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
