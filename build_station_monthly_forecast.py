"""Build monthly 2026 station forecasts for the Carbon Metrics experience."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

DATA_FILES = sorted(Path("data").glob("[0-9]*.csv"))
OUTPUT = Path("output/station-monthly-forecast.json")
FORECAST_YEAR = 2026
FORECAST_MONTHS = [8, 9, 10, 11, 12]


def records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def month_key(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


def load_orders() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.concat((pd.read_csv(path, low_memory=False) for path in DATA_FILES), ignore_index=True)
    raw["booking_date"] = pd.to_datetime(raw["booking_date"], errors="coerce", format="mixed")
    raw = raw.dropna(subset=["booking_date", "order_id"]).copy()
    for column, fallback in [("station_code", "UNKNOWN"), ("station_name", "Unknown station"), ("item_name", "Unknown item"), ("order_status", "Unknown")]:
        raw[column] = raw[column].fillna(fallback).astype(str).str.strip().replace("", fallback)
    raw["quantity"] = pd.to_numeric(raw["quantity"], errors="coerce").fillna(0)
    raw["amount_payable"] = pd.to_numeric(raw["amount_payable"], errors="coerce")
    raw["gross_item_amount"] = pd.to_numeric(raw["gross_item_amount"], errors="coerce").fillna(0)
    raw["year"] = raw.booking_date.dt.year
    raw["monthNumber"] = raw.booking_date.dt.month
    raw["month"] = raw.booking_date.dt.to_period("M").astype(str)
    raw["weekOfMonth"] = ((raw.booking_date.dt.day - 1) // 7 + 1).astype(int)
    raw["is_cancelled"] = raw.order_status.str.contains("cancel", case=False, na=False)
    raw["is_fulfilled"] = raw.order_status.isin(["Delivered", "Partially Delivered"])

    orders = raw.groupby("order_id", as_index=False).agg(
        bookingDate=("booking_date", "first"), stationCode=("station_code", "first"), stationName=("station_name", "first"), year=("year", "first"),
        monthNumber=("monthNumber", "first"), month=("month", "first"), isCancelled=("is_cancelled", "max"),
        revenue=("amount_payable", "max"), fallbackRevenue=("gross_item_amount", "sum"),
    )
    orders["revenue"] = orders.revenue.fillna(orders.fallbackRevenue).fillna(0)
    orders["weekOfMonth"] = ((orders.bookingDate.dt.day - 1) // 7 + 1).astype(int)
    return raw, orders


def xgboost_model() -> Pipeline:
    processor = ColumnTransformer([
        ("station", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), ["stationCode"]),
        ("number", SimpleImputer(strategy="median"), ["monthNumber", "yearIndex"]),
    ])
    regressor = XGBRegressor(objective="reg:squarederror", n_estimators=180, max_depth=4, learning_rate=0.05, subsample=0.85, colsample_bytree=0.9, random_state=42, n_jobs=4)
    return Pipeline([("prepare", processor), ("model", regressor)])


def confidence(history: pd.DataFrame) -> int:
    observations = history.year.nunique()
    volatility = history.orderCount.std(ddof=0) / max(history.orderCount.mean(), 1)
    return int(max(55, min(94, 56 + observations * 13 - volatility * 12)))


def top_items(raw: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    fulfilled = raw[raw.is_fulfilled & raw.monthNumber.isin(FORECAST_MONTHS)].copy()
    actual = fulfilled.groupby(["station_code", "station_name", "year", "monthNumber", "item_name"], as_index=False).agg(units=("quantity", "sum"))
    actual = actual.sort_values(["station_code", "year", "monthNumber", "units"], ascending=[True, True, True, False])
    actual["rank"] = actual.groupby(["station_code", "year", "monthNumber"]).cumcount() + 1
    actual = actual[actual["rank"] <= 10].rename(columns={"station_code": "stationCode", "station_name": "stationName", "item_name": "item"})

    forecast_rows: list[dict[str, object]] = []
    for summary in summaries.itertuples(index=False):
        previous = actual[(actual.stationCode == summary.stationCode) & (actual.year == 2025) & (actual.monthNumber == summary.monthNumber)]
        if previous.empty:
            previous = actual[(actual.stationCode == summary.stationCode) & (actual.monthNumber == summary.monthNumber)].sort_values("year", ascending=False).head(10)
        reference_orders = max(float(summary.referenceOrders), 1)
        for item in previous.itertuples(index=False):
            forecast_rows.append({
                "stationCode": summary.stationCode, "stationName": summary.stationName, "year": FORECAST_YEAR,
                "monthNumber": summary.monthNumber, "month": summary.month, "item": item.item,
                "units": round(item.units * summary.forecastOrders / reference_orders), "rank": item.rank, "isForecast": True,
            })
    actual["isForecast"] = False
    return pd.concat([actual, pd.DataFrame(forecast_rows)], ignore_index=True)


def main() -> None:
    raw, orders = load_orders()
    station_month = orders.groupby(["stationCode", "stationName", "year", "monthNumber", "month"], as_index=False).agg(
        orderCount=("order_id", "size"), cancelledOrders=("isCancelled", "sum"), revenue=("revenue", "sum"),
        revenueLoss=("revenue", lambda values: values[orders.loc[values.index, "isCancelled"]].sum()),
    )
    station_month["cancelRate"] = station_month.cancelledOrders / station_month.orderCount.clip(lower=1)
    station_month["yearIndex"] = station_month.year - station_month.year.min()
    weekly_history = orders.groupby(["stationCode", "year", "monthNumber", "weekOfMonth"], as_index=False).agg(
        orderCount=("order_id", "size"), cancelledOrders=("isCancelled", "sum"), revenue=("revenue", "sum"),
        revenueLoss=("revenue", lambda values: values[orders.loc[values.index, "isCancelled"]].sum()),
    )
    weekly_item_history = (
        raw[raw.is_fulfilled & raw.monthNumber.isin(FORECAST_MONTHS)]
        .groupby(["station_code", "year", "monthNumber", "weekOfMonth", "item_name"], as_index=False)
        .agg(units=("quantity", "sum"))
    )
    def weekly_item_lookup(frame: pd.DataFrame) -> dict[tuple[str, int, int], list[dict[str, object]]]:
        grouped = frame.groupby(["station_code", "monthNumber", "weekOfMonth", "item_name"], as_index=False).units.sum()
        return {
            (station, int(month), int(week)): records(group.sort_values("units", ascending=False).head(10).rename(columns={"item_name": "item"})[["item", "units"]])
            for (station, month, week), group in grouped.groupby(["station_code", "monthNumber", "weekOfMonth"])
        }
    weekly_items_2025 = weekly_item_lookup(weekly_item_history[weekly_item_history.year.eq(2025)])
    weekly_items_all = weekly_item_lookup(weekly_item_history)
    weekly_order_counts_2025 = orders[orders.year.eq(2025)].groupby(["stationCode", "monthNumber", "weekOfMonth"]).size().to_dict()
    weekly_order_counts_all = orders.groupby(["stationCode", "monthNumber", "weekOfMonth"]).size().to_dict()
    train = station_month[station_month.monthNumber.isin(FORECAST_MONTHS)].copy()
    model = xgboost_model()
    model.fit(train[["stationCode", "monthNumber", "yearIndex"]], train.orderCount)

    station_lookup = train[["stationCode", "stationName"]].drop_duplicates().sort_values("stationCode")
    targets = pd.MultiIndex.from_product([station_lookup.stationCode, FORECAST_MONTHS], names=["stationCode", "monthNumber"]).to_frame(index=False)
    targets = targets.merge(station_lookup, on="stationCode", how="left")
    targets["yearIndex"] = FORECAST_YEAR - int(train.year.min())
    targets["month"] = targets.monthNumber.map(lambda number: month_key(FORECAST_YEAR, int(number)))
    targets["xgbOrders"] = model.predict(targets[["stationCode", "monthNumber", "yearIndex"]]).clip(0)

    output_rows: list[dict[str, object]] = []
    for target in targets.itertuples(index=False):
        history = train[(train.stationCode == target.stationCode) & (train.monthNumber == target.monthNumber)].sort_values("year")
        if history.empty:
            continue
        reference = history[history.year == 2025]
        reference_orders = float(reference.orderCount.iloc[0]) if not reference.empty else float(history.orderCount.iloc[-1])
        seasonal = history.orderCount.tail(3).mean()
        forecast_orders = round(max(0, 0.70 * seasonal + 0.30 * target.xgbOrders))
        cancellation_rate = float(history.cancelRate.mean())
        average_value = float((history.revenue / history.orderCount.clip(lower=1)).mean())
        forecast_revenue = round(forecast_orders * average_value)
        forecast_cancelled = round(forecast_orders * cancellation_rate)
        revenue_loss = round(forecast_revenue * cancellation_rate)
        output_rows.append({
            "stationCode": target.stationCode, "stationName": target.stationName, "month": target.month,
            "monthNumber": int(target.monthNumber), "forecastOrders": forecast_orders,
            "predictedPercentage": round((forecast_orders / max(reference_orders, 1) - 1) * 100, 1),
            "forecastCancelledOrders": forecast_cancelled, "forecastRevenueLoss": revenue_loss,
            "predictedRevenue": forecast_revenue, "confidenceScore": confidence(history),
            "referenceOrders": round(reference_orders), "historicalYears": int(history.year.nunique()),
            "averageCancellationRate": round(cancellation_rate * 100, 1),
        })
    summaries = pd.DataFrame(output_rows).sort_values(["month", "forecastOrders"], ascending=[True, False])
    items = top_items(raw, summaries)

    details: list[dict[str, object]] = []
    for forecast in summaries.itertuples(index=False):
        historical = station_month[(station_month.stationCode == forecast.stationCode) & (station_month.monthNumber == forecast.monthNumber) & (station_month.year.isin([2022, 2023, 2024, 2025]))].copy()
        trend = [
            {"year": int(row.year), "orders": int(row.orderCount), "cancelledOrders": int(row.cancelledOrders), "revenue": int(row.revenue), "revenueLoss": int(row.revenueLoss), "isForecast": False}
            for row in historical.sort_values("year").itertuples(index=False)
        ]
        trend.append({"year": FORECAST_YEAR, "orders": int(forecast.forecastOrders), "cancelledOrders": int(forecast.forecastCancelledOrders), "revenue": int(forecast.predictedRevenue), "revenueLoss": int(forecast.forecastRevenueLoss), "isForecast": True})
        top_items_by_year = []
        for year in [2022, 2023, 2024, 2025, FORECAST_YEAR]:
            ranked = items[(items.stationCode == forecast.stationCode) & (items.monthNumber == forecast.monthNumber) & (items.year == year)].sort_values("rank")
            top_items_by_year.append({"year": year, "isForecast": year == FORECAST_YEAR, "items": records(ranked[["rank", "item", "units"]])})
        weekly = weekly_history[(weekly_history.stationCode == forecast.stationCode) & (weekly_history.monthNumber == forecast.monthNumber) & (weekly_history.year == 2025)].copy()
        if weekly.empty:
            weekly = weekly_history[(weekly_history.stationCode == forecast.stationCode) & (weekly_history.monthNumber == forecast.monthNumber)].copy()
        target_start = pd.Timestamp(forecast.month + "-01")
        target_end = target_start + pd.offsets.MonthEnd(0)
        target_weeks = list(range(1, ((target_end.day - 1) // 7 + 1) + 1))
        week_shares = weekly.groupby("weekOfMonth").orderCount.sum().reindex(target_weeks, fill_value=0).astype(float)
        if week_shares.sum() == 0:
            week_shares[:] = 1
        week_shares = week_shares / week_shares.sum()
        weekly_rows = []
        for week, share in week_shares.items():
            start = target_start + pd.Timedelta(days=(week - 1) * 7)
            end = min(start + pd.Timedelta(days=6), target_end)
            weekly_key = (forecast.stationCode, int(forecast.monthNumber), int(week))
            item_ranking = weekly_items_2025.get(weekly_key, weekly_items_all.get(weekly_key, []))
            item_reference_orders = weekly_order_counts_2025.get(weekly_key, weekly_order_counts_all.get(weekly_key, 0))
            weekly_orders = round(forecast.forecastOrders * share)
            weekly_rows.append({
                "week": int(week), "label": f"Week {week} · {start.strftime('%d %b')} - {end.strftime('%d %b')}",
                "forecastOrders": weekly_orders, "forecastCancelledOrders": round(forecast.forecastCancelledOrders * share),
                "predictedRevenue": round(forecast.predictedRevenue * share), "forecastRevenueLoss": round(forecast.forecastRevenueLoss * share),
                "topItems": [
                    {"rank": rank, "item": item["item"], "units": round(item["units"] * weekly_orders / max(item_reference_orders, 1))}
                    for rank, item in enumerate(item_ranking, start=1)
                ],
            })
        # Rounding must not make the weekly total disagree with the card total.
        if weekly_rows:
            weekly_rows[0]["forecastOrders"] += int(forecast.forecastOrders) - sum(row["forecastOrders"] for row in weekly_rows)
            weekly_rows[0]["forecastCancelledOrders"] += int(forecast.forecastCancelledOrders) - sum(row["forecastCancelledOrders"] for row in weekly_rows)
            weekly_rows[0]["predictedRevenue"] += int(forecast.predictedRevenue) - sum(row["predictedRevenue"] for row in weekly_rows)
            weekly_rows[0]["forecastRevenueLoss"] += int(forecast.forecastRevenueLoss) - sum(row["forecastRevenueLoss"] for row in weekly_rows)
        details.append({
            "stationCode": forecast.stationCode, "stationName": forecast.stationName, "month": forecast.month,
            "forecastOrders": int(forecast.forecastOrders), "predictedPercentage": float(forecast.predictedPercentage),
            "forecastCancelledOrders": int(forecast.forecastCancelledOrders), "forecastRevenueLoss": int(forecast.forecastRevenueLoss),
            "predictedRevenue": int(forecast.predictedRevenue), "confidenceScore": int(forecast.confidenceScore),
            "trend": trend, "topItemsByYear": top_items_by_year, "weeklyForecast": weekly_rows,
            "drivers": [
                f"{forecast.historicalYears} years of same-month station history",
                f"2025 reference volume: {int(forecast.referenceOrders):,} orders",
                f"Historical cancellation rate: {forecast.averageCancellationRate}%",
                "Station code, month and historical demand are used in the XGBoost trend estimate",
                "Weekly forecast is allocated from the selected monthly prediction using the station's historical same-month weekly demand share",
            ],
        })

    payload = {
        "metadata": {"model": "XGBoost monthly station-demand forecast", "forecastYear": FORECAST_YEAR, "months": [month_key(FORECAST_YEAR, month) for month in FORECAST_MONTHS], "note": "2026 cancellation and revenue-loss values are forecast estimates based on historical station-month cancellation behaviour."},
        "stationSummaries": records(summaries), "stationDetails": details,
    }
    OUTPUT.write_text(json.dumps(payload, allow_nan=False))
    print(json.dumps({"output": str(OUTPUT), "stations": int(summaries.stationCode.nunique()), "monthlyCards": len(summaries), "details": len(details)}, indent=2))


if __name__ == "__main__":
    main()
