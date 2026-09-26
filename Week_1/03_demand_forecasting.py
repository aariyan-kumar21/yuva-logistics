"""
03_demand_forecasting.py
-------------------------
Stage 3 of the strategic roadmap: Predictive Modelling — Demand Forecasting.

Illustrates how a regression-based approach can forecast daily order volume
per warehouse, which feeds into inventory replenishment planning (reducing
stockouts and overstock, and directly improving the Inventory Turnover KPI).

Model used: Linear Regression on engineered time features, benchmarked
against a Random Forest Regressor. In production this would be replaced
with a proper time-series model (e.g. SARIMA, Prophet, or a gradient-boosted
model with lag features) once enough historical data is available.

Run: python 03_demand_forecasting.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.dirname(__file__))
from importlib import import_module
_dc = import_module("01_data_cleaning")


def build_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby([df["order_date"].dt.date, "warehouse_id"])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={"order_date": "date", "quantity": "units_ordered"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_of_month"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)
    return daily


def train_and_evaluate(daily: pd.DataFrame):
    daily = pd.get_dummies(daily, columns=["warehouse_id"], drop_first=True)

    feature_cols = [c for c in daily.columns if c not in ("date", "units_ordered")]
    X = daily[feature_cols]
    y = daily["units_ordered"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "MAE": round(mean_absolute_error(y_test, preds), 2),
            "R2": round(r2_score(y_test, preds), 3),
        }

    return results


if __name__ == "__main__":
    raw = _dc.load_data()
    cleaned = _dc.clean_orders(raw)

    daily_demand = build_daily_demand(cleaned)
    print(f"Built daily demand table with {len(daily_demand)} warehouse-day rows.")

    metrics = train_and_evaluate(daily_demand)
    print("\nModel comparison for next-day demand forecasting:")
    for model_name, m in metrics.items():
        print(f"  {model_name}: MAE={m['MAE']} units, R2={m['R2']}")

    print(
        "\nInterpretation: MAE indicates the average forecast error in units per "
        "warehouse-day; R2 indicates how much variance in daily demand the model "
        "explains. These forecasts would feed a reorder-point / safety-stock "
        "calculation for each warehouse-SKU combination."
    )
