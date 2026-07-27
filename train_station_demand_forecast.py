"""Daily station-demand XGBoost forecast and operational-readiness dataset.

This script is intentionally conservative: it trains only on information that is
available before the forecast day. Route forecasts are allocated from each
station's XGBoost forecast using recent train-route order share, so they are
labelled as derived route forecasts rather than a second independent model.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_FILES = sorted(Path("data").glob("[0-9]*.csv"))
EVENTS_FILE = Path("data/calendar_events.csv")
OUTPUT = Path("output/station-demand-dashboard.json")
FEATURES = [
    "station_code", "month", "day_of_week", "is_weekend", "festival_flag", "holiday_flag",
    "local_event_flag", "lag_1_demand", "rolling_7_demand", "vendor_capacity",
]
FORECAST_DAYS = 30


def events() -> pd.DataFrame:
    """Load known calendar signals and collapse multiple events on the same day."""
    columns = ["date", "event_name", "event_type", "festival_flag", "holiday_flag", "local_event_flag"]
    if not EVENTS_FILE.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(EVENTS_FILE)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", format="mixed").dt.normalize()
    for column in ["festival_flag", "holiday_flag", "local_event_flag"]:
        frame[column] = pd.to_numeric(frame.get(column, 0), errors="coerce").fillna(0).clip(0, 1).astype(int)
    if "event_name" not in frame:
        frame["event_name"] = "Known calendar event"
    if "event_type" not in frame:
        frame["event_type"] = "Holiday"
    frame["event_name"] = frame["event_name"].fillna("Known calendar event").astype(str)
    frame["event_type"] = frame["event_type"].fillna("Holiday").astype(str)
    return (
        frame.dropna(subset=["date"])
        .groupby("date", as_index=False)
        .agg(
            event_name=("event_name", lambda values: " / ".join(dict.fromkeys(values))),
            event_type=("event_type", lambda values: " / ".join(dict.fromkeys(values))),
            festival_flag=("festival_flag", "max"),
            holiday_flag=("holiday_flag", "max"),
            local_event_flag=("local_event_flag", "max"),
        )
    )


def order_rows() -> pd.DataFrame:
    raw = pd.concat((pd.read_csv(path, low_memory=False) for path in DATA_FILES), ignore_index=True)
    raw["delivery_date"] = pd.to_datetime(raw["delivery_date"], errors="coerce", format="mixed")
    raw = raw[raw["order_status"].fillna("").isin(["Delivered", "Partially Delivered"])].dropna(subset=["order_id", "delivery_date"]).copy()
    raw["date"] = raw["delivery_date"].dt.normalize()
    for column, fallback in [("station_code", "UNKNOWN"), ("station_name", "Unknown station"), ("outlet_name", "Unknown outlet"), ("train_no", "Unknown"), ("train_name", "Unknown route")]:
        raw[column] = raw[column].fillna(fallback).astype(str).str.strip().replace("", fallback)
    raw["amount_payable"] = pd.to_numeric(raw["amount_payable"], errors="coerce")
    raw["gross_item_amount"] = pd.to_numeric(raw["gross_item_amount"], errors="coerce").fillna(0)

    # The source export repeats an order once per food item. Deduplicate before
    # aggregation to avoid inflating both volume and revenue.
    orders = raw.groupby("order_id", as_index=False).agg(
        date=("date", "first"), station_code=("station_code", "first"), station_name=("station_name", "first"),
        outlet_name=("outlet_name", "first"), train_no=("train_no", "first"), train_name=("train_name", "first"),
        revenue=("amount_payable", "max"), fallback_revenue=("gross_item_amount", "sum"),
    )
    orders["revenue"] = orders.revenue.fillna(orders.fallback_revenue).fillna(0)
    return orders


def daily_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return orders.groupby(["station_code", "station_name", "date"], as_index=False).agg(
        order_count=("order_id", "size"), revenue=("revenue", "sum"), vendor_capacity=("outlet_name", "nunique")
    ).sort_values(["station_code", "date"])


def with_features(daily: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    frame = daily.merge(calendar, on="date", how="left")
    for column in ["festival_flag", "holiday_flag", "local_event_flag"]:
        frame[column] = frame[column].fillna(0)
    frame["month"] = frame.date.dt.month
    frame["day_of_week"] = frame.date.dt.dayofweek
    frame["is_weekend"] = frame.day_of_week.ge(5).astype(int)
    frame["new_segment"] = frame.groupby("station_code").date.diff().dt.days.ne(1).fillna(True).astype(int)
    frame["segment"] = frame.groupby("station_code").new_segment.cumsum()
    groups = frame.groupby(["station_code", "segment"], group_keys=False)
    frame["lag_1_demand"] = groups.order_count.shift(1)
    frame["rolling_7_demand"] = groups.order_count.transform(lambda values: values.shift(1).rolling(7, min_periods=2).mean())
    return frame


def make_model() -> Pipeline:
    processor = ColumnTransformer([
        ("station", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=3))]), ["station_code"]),
        ("numeric", SimpleImputer(strategy="median"), FEATURES[1:]),
    ])
    regressor = XGBRegressor(
        objective="reg:squarederror", n_estimators=250, max_depth=6, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.85, min_child_weight=4, random_state=42, n_jobs=4,
    )
    return Pipeline([("prepare", processor), ("model", regressor)])


def risk_level(gap_share: float) -> str:
    if gap_share >= 0.30:
        return "Critical"
    if gap_share >= 0.18:
        return "High"
    if gap_share >= 0.06:
        return "Medium"
    return "Normal"


def future_forecast(fitted: Pipeline, daily: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    latest = daily.date.max()
    capacity = daily.assign(orders_per_vendor=daily.order_count / daily.vendor_capacity.clip(lower=1)).groupby(["station_code", "station_name"], as_index=False).agg(
        current_vendor_count=("vendor_capacity", "median"), orders_per_vendor=("orders_per_vendor", "median"),
        revenue=("revenue", "sum"), orders=("order_count", "sum"), baseline_daily_orders=("order_count", lambda values: values.tail(28).mean()),
    )
    capacity["average_order_value"] = capacity.revenue / capacity.orders.clip(lower=1)
    histories = {code: list(group.sort_values("date").tail(7).order_count.astype(float)) for code, group in daily.groupby("station_code")}
    rows: list[dict[str, object]] = []

    for date in pd.date_range(latest + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq="D"):
        matching_event = calendar[calendar.date.eq(date)]
        event = matching_event.iloc[0].to_dict() if not matching_event.empty else {}
        candidates = []
        for station in capacity.itertuples(index=False):
            history = histories[station.station_code]
            candidates.append({
                "station_code": station.station_code, "month": date.month, "day_of_week": date.dayofweek,
                "is_weekend": int(date.dayofweek >= 5), "festival_flag": event.get("festival_flag", 0),
                "holiday_flag": event.get("holiday_flag", 0), "local_event_flag": event.get("local_event_flag", 0),
                "lag_1_demand": history[-1], "rolling_7_demand": sum(history) / len(history), "vendor_capacity": station.current_vendor_count,
            })
        predictions = fitted.predict(pd.DataFrame(candidates)[FEATURES]).clip(0)
        for station, prediction in zip(capacity.itertuples(index=False), predictions, strict=True):
            predicted = float(prediction)
            history = histories[station.station_code]
            history.append(predicted)
            histories[station.station_code] = history[-7:]
            prepared_capacity = max(station.baseline_daily_orders * 1.10, 1)
            gap = max(predicted - prepared_capacity, 0)
            gap_share = gap / max(predicted, 1)
            recommended = max(1, round(predicted / max(station.orders_per_vendor, 1)))
            rows.append({
                "stationCode": station.station_code, "stationName": station.station_name, "date": date.strftime("%Y-%m-%d"),
                "forecastOrders": round(predicted), "forecastRevenue": round(predicted * station.average_order_value),
                "revenueAtRisk": round(gap * station.average_order_value), "currentVendorCount": round(station.current_vendor_count),
                "recommendedVendorCount": recommended, "vendorGap": max(recommended - round(station.current_vendor_count), 0),
                "baselineDailyOrders": round(station.baseline_daily_orders), "readinessScore": round(max(20, min(100, 100 - gap_share * 200))),
                "riskLevel": risk_level(gap_share), "festival": bool(event.get("festival_flag", 0)), "holiday": bool(event.get("holiday_flag", 0)),
                "eventName": event.get("event_name") or None, "eventType": event.get("event_type") or None,
            })
    return pd.DataFrame(rows)


def station_summary(daily_forecast: pd.DataFrame) -> pd.DataFrame:
    summary = daily_forecast.groupby(["stationCode", "stationName"], as_index=False).agg(
        forecastOrders=("forecastOrders", "sum"), forecastRevenue=("forecastRevenue", "sum"),
        revenueAtRisk=("revenueAtRisk", "sum"), recommendedVendorCount=("recommendedVendorCount", "max"),
        currentVendorCount=("currentVendorCount", "max"), vendorGap=("vendorGap", "max"),
        baselineDailyOrders=("baselineDailyOrders", "max"), readinessScore=("readinessScore", "min"),
    )
    summary["demandUpliftPercent"] = ((summary.forecastOrders / (summary.baselineDailyOrders * FORECAST_DAYS).clip(lower=1) - 1) * 100).round(1)
    summary["riskSharePercent"] = ((summary.revenueAtRisk / summary.forecastRevenue.clip(lower=1)) * 100).round(1)
    summary["riskLevel"] = summary.riskSharePercent.map(lambda share: risk_level(float(share) / 100))
    summary["recommendedAction"] = summary.apply(
        lambda row: "Maintain normal operations and monitor live demand." if row.riskLevel == "Normal" else
        f"Add {int(row.vendorGap)} vendor(s), stage inventory and review the next peak window.", axis=1,
    )
    return summary.sort_values(["riskLevel", "forecastOrders"], ascending=[True, False])


def route_forecasts(orders: pd.DataFrame, daily_forecast: pd.DataFrame) -> pd.DataFrame:
    """Allocate station forecast volume to the routes that recently served it."""
    recent_start = orders.date.max() - pd.Timedelta(days=90)
    recent = orders[orders.date.ge(recent_start)].copy()
    routes = recent.groupby(["station_code", "station_name", "train_no", "train_name"], as_index=False).agg(recentOrders=("order_id", "size"))
    routes = routes[routes.recentOrders.ge(2)]
    routes["routeShare"] = routes.recentOrders / routes.groupby("station_code").recentOrders.transform("sum")
    routes = routes.sort_values(["station_code", "routeShare"], ascending=[True, False]).groupby("station_code").head(5)
    forecasts = daily_forecast.merge(routes, left_on=["stationCode", "stationName"], right_on=["station_code", "station_name"], how="inner")
    output = forecasts.groupby(["stationCode", "stationName", "train_no", "train_name"], as_index=False).agg(
        forecastOrders=("forecastOrders", lambda values: round((values * forecasts.loc[values.index, "routeShare"]).sum())),
        forecastRevenue=("forecastRevenue", lambda values: round((values * forecasts.loc[values.index, "routeShare"]).sum())),
        revenueAtRisk=("revenueAtRisk", lambda values: round((values * forecasts.loc[values.index, "routeShare"]).sum())),
        routeShare=("routeShare", "first"),
    )
    output = output.rename(columns={"train_no": "trainNo", "train_name": "trainName"})
    output["routeSharePercent"] = (output.routeShare * 100).round(1)
    output["riskLevel"] = output.apply(lambda row: risk_level(row.revenueAtRisk / max(row.forecastRevenue, 1)), axis=1)
    return output.sort_values(["stationCode", "forecastOrders"], ascending=[True, False])


def event_alerts(daily_forecast: pd.DataFrame, forecast_start: pd.Timestamp) -> list[dict[str, object]]:
    event_days = daily_forecast[daily_forecast.eventName.notna()].groupby(["date", "eventName", "eventType"], as_index=False).agg(
        forecastOrders=("forecastOrders", "sum"), revenueAtRisk=("revenueAtRisk", "sum"),
    )
    normal_average = daily_forecast[~daily_forecast.eventName.notna()].groupby("date", as_index=False).forecastOrders.sum().forecastOrders.mean()
    alerts = []
    for event in event_days.sort_values("date").itertuples(index=False):
        event_date = pd.Timestamp(event.date)
        alerts.append({
            "date": event.date, "eventName": event.eventName, "eventType": event.eventType,
            "daysUntil": max((event_date - forecast_start).days, 0), "forecastOrders": int(event.forecastOrders),
            "expectedDemandChangePercent": round((event.forecastOrders / max(normal_average, 1) - 1) * 100, 1),
            "revenueAtRisk": int(event.revenueAtRisk),
        })
    return alerts


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Return JSON-safe records; pandas otherwise serialises empty object cells as NaN."""
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def main() -> None:
    orders = order_rows()
    daily = daily_orders(orders)
    calendar = events()
    frame = with_features(daily, calendar).dropna(subset=["lag_1_demand", "rolling_7_demand"])
    cutoff = frame.date.max() - pd.Timedelta(days=30)
    train, test = frame[frame.date.le(cutoff)], frame[frame.date.gt(cutoff)]
    if len(train) < 100 or test.empty:
        raise ValueError("Need at least 100 train rows and a 30-day holdout.")
    fitted = make_model()
    fitted.fit(train[FEATURES], train.order_count)
    prediction = fitted.predict(test[FEATURES]).clip(0)
    evaluation = {
        "testStart": test.date.min().strftime("%Y-%m-%d"), "testEnd": test.date.max().strftime("%Y-%m-%d"),
        "rmse": round(float(mean_squared_error(test.order_count, prediction) ** 0.5), 2),
        "mae": round(float(mean_absolute_error(test.order_count, prediction)), 2), "r2": round(float(r2_score(test.order_count, prediction)), 4),
        "trainingRows": len(train), "testRows": len(test),
    }
    fitted.fit(frame[FEATURES], frame.order_count)
    daily_forecast = future_forecast(fitted, daily, calendar)
    station_forecasts = station_summary(daily_forecast)
    routes = route_forecasts(orders, daily_forecast)
    forecast_start = pd.Timestamp(daily_forecast.date.min())
    payload = {
        "metadata": {
            "model": "XGBoost daily station-demand forecast", "forecastStart": daily_forecast.date.min(),
            "forecastEnd": daily_forecast.date.max(), "forecastDays": FORECAST_DAYS,
            "routeForecastMethod": "Station XGBoost demand forecast allocated using each station's trailing 90-day train-route order share.",
        },
        "evaluation": evaluation, "stationForecasts": records(station_forecasts),
        "dailyForecasts": records(daily_forecast), "routeForecasts": records(routes),
        "eventAlerts": event_alerts(daily_forecast, forecast_start),
    }
    OUTPUT.write_text(json.dumps(payload, allow_nan=False, default=lambda value: int(value) if hasattr(value, "item") else str(value)))
    print(json.dumps({"output": str(OUTPUT), "stations": len(station_forecasts), "routes": len(routes), "alerts": len(payload["eventAlerts"]), "evaluation": evaluation}, indent=2))


if __name__ == "__main__":
    main()
