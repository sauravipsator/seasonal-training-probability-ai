# Notebook guide

Run the notebooks in order from this directory after activating the project virtual environment.

1. `01_data_cleaning.ipynb` validates the raw line-item exports and deduplicates orders.
2. `02_feature_engineering.ipynb` shows the time-safe model inputs.
3. `03_eda.ipynb` explores demand, seasonality and train-route concentration.
4. `04_model_training.ipynb` executes the reproducible station forecast and inspects the dashboard output.

The notebooks deliberately do not auto-save models. The production-style command is still `python train_station_demand_forecast.py`, which writes a fresh `output/station-demand-dashboard.json` for the Java API and Carbon dashboard.
