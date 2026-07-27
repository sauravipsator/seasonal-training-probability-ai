# AI Seasonal Demand Forecasting

A reproducible operations prototype that turns historical food-order exports into demand forecasts, vendor-readiness actions, event alerts and train-route planning signals.

```text
Historical orders + calendar events
        -> cleaning and order deduplication
        -> time-safe feature engineering
        -> XGBoost daily station-demand forecast
        -> readiness, risk, route allocation and event alerts
        -> Java API + Carbon dashboard
```

## What the demo shows

- 30-day daily demand and revenue forecast for each delivery station
- Readiness score plus Normal, Medium, High or Critical capacity risk
- Recommended vendor count and the number of additional vendors needed
- Festival and holiday countdown alerts from a populated calendar
- Top train-route forecasts, transparently derived from each station forecast and the most recent 90-day train demand share
- Network-wide food-demand ranking and cancellation-risk model
- A responsive Carbon metrics page and station drill-down page

The dashboard gives preparation guidance, not a guarantee. Operations must confirm live vendor availability, weather, train disruptions, local events and final holiday dates before action.

## Project layout

```text
xgboost/
├── data/
│   ├── 2023.csv ... 2025-4.csv       Historical line-item order exports
│   └── calendar_events.csv            Festivals, holidays and local signals
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda.ipynb
│   └── 04_model_training.ipynb
├── output/
│   └── station-demand-dashboard.json Dashboard-ready model output
├── reports/
│   └── EDA_Report.pdf                 Demo-ready EDA and model report
├── dashboard-backend/                 Spring Boot API, port 8090 by default
├── train_station_demand_forecast.py   Daily station forecast and readiness engine
├── train_forecast.py                  Network-wide monthly food ranking
├── train_cancellation_model.py        Booking-time cancellation-risk model
└── generate_eda_report.py             Builds the PDF report from model output
```

## Quick start

Requirements: Python 3.10+, Java 17+ and Maven. The supplied virtual environment can be used if it already exists.

```bash
cd /Users/sauravkumar/Documents/xgboost
source .venv/bin/activate
python -m pip install -r requirements.txt

# Train the supporting models and dashboard payloads.
python train_forecast.py --input data/2023.csv data/2024.csv data/2025-1.csv data/2025-2.csv data/2025-3.csv data/2025-4.csv
python train_cancellation_model.py
python build_dashboard_data.py
python train_station_demand_forecast.py
python generate_eda_report.py

# Run the API from the folder that contains pom.xml.
cd dashboard-backend
mvn spring-boot:run
```

Open the standalone API dashboard at [http://localhost:8090](http://localhost:8090). The Carbon frontend reads the same service through `XGBOOST_DASHBOARD_URL=http://127.0.0.1:8090`.

## Model and validation

`train_station_demand_forecast.py` removes cancelled orders, deduplicates repeated food-item rows by `order_id`, and aggregates fulfilled orders by station and delivery day. It uses only information known before a prediction day:

- station code, month, weekday and weekend
- festival, holiday and local-event flags
- previous-day demand and 7-day moving average
- observed vendor capacity

The model is evaluated with a chronological 30-day holdout. The current generated payload reports its exact MAE, RMSE and R-squared in `output/station-demand-dashboard.json`; do not replace those values with a generic accuracy claim.

### Readiness and risk policy

Prepared capacity is the recent daily baseline plus a 10% operating buffer. The model converts the gap between prediction and this capacity into revenue at risk, a 20-100 readiness score and a risk level:

| Risk | Capacity-gap share |
| --- | --- |
| Normal | under 6% |
| Medium | 6% to under 18% |
| High | 18% to under 30% |
| Critical | 30% or more |

## Calendar events

`data/calendar_events.csv` contains widely observed Indian festivals and national holidays through 2027, plus an example local travel-peak signal. It is deliberately editable. Add station- or route-specific events as new rows and set the relevant flag. Verify dates with the official/local operating calendar before every production run, especially for lunar festivals.

## API endpoints

After starting Spring Boot:

| Endpoint | Purpose |
| --- | --- |
| `/api/station-demand/summary` | Station readiness cards |
| `/api/station-demand/station?stationCode=NDLS` | Daily station forecast |
| `/api/station-demand/alerts` | Festival and holiday countdown alerts |
| `/api/station-demand/routes?stationCode=NDLS` | Station train-route forecasts |
| `/api/forecast` | Monthly food-demand ranking |
| `/api/vendor-summary` | Local/Gemini narrative summary |

## Demo flow

1. Start with the Carbon Metrics page: show total station demand and the 30-day comparison chart.
2. Call out a festival alert and the readiness/risk label; explain that it is capacity guidance, not a promise.
3. Open a station: show the daily forecast, vendor requirement, event countdown and top train routes.
4. Use the route card to explain where the station demand is expected to come from.
5. Close with the EDA report and chronological evaluation metrics to show the model is measured, reproducible and connected to actions.

## Notebooks and report

See [notebooks/README.md](notebooks/README.md) for the four notebook walkthrough. Regenerate the PDF after every model run:

```bash
python generate_eda_report.py
```

The report is written to `reports/EDA_Report.pdf`.
