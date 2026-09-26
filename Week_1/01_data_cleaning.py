"""
01_data_cleaning.py
--------------------
Stage 1 of the strategic roadmap: Data Collection & Cleaning.

Ingests raw order/shipment records (from a WMS/TMS export in production,
or a synthetic sample here), and produces a cleaned, analysis-ready table:
- parses dates
- removes duplicate order records
- imputes/flags missing values
- derives basic fields needed downstream (cycle time, on-time flag)

Run: python 01_data_cleaning.py
"""

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def generate_synthetic_orders(n=5000):
    """Creates a realistic synthetic order/shipment dataset for a 3-warehouse
    e-commerce fulfillment network, standing in for a WMS/TMS export."""
    warehouses = ["WH_NORTH", "WH_CENTRAL", "WH_SOUTH"]
    zones = [f"ZONE_{i}" for i in range(1, 13)]

    order_date = pd.to_datetime("2025-01-01") + pd.to_timedelta(
        RNG.integers(0, 270, n), unit="D"
    )
    processing_days = RNG.gamma(shape=2.0, scale=0.6, size=n)  # warehouse handling
    transit_days = RNG.gamma(shape=2.5, scale=0.9, size=n)     # carrier transit
    promised_days = np.full(n, 3.0)                            # 3-day SLA promise

    delivered_offset = processing_days + transit_days
    delivered_date = order_date + pd.to_timedelta(delivered_offset, unit="D")
    promised_date = order_date + pd.to_timedelta(promised_days, unit="D")

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "warehouse_id": RNG.choice(warehouses, n, p=[0.4, 0.35, 0.25]),
        "dest_zone": RNG.choice(zones, n),
        "order_date": order_date,
        "promised_date": promised_date,
        "delivered_date": delivered_date,
        "quantity": RNG.integers(1, 8, n),
        "unit_cost": np.round(RNG.uniform(5, 120, n), 2),
        "shipping_cost": np.round(RNG.uniform(3, 25, n), 2),
    })

    # inject some realistic data-quality issues to clean
    dupe_idx = RNG.choice(df.index, size=int(0.01 * n), replace=False)
    df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)          # duplicates
    missing_idx = RNG.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_idx, "shipping_cost"] = np.nan                       # missing values
    neg_idx = RNG.choice(df.index, size=int(0.005 * len(df)), replace=False)
    df.loc[neg_idx, "quantity"] = -1                                    # bad values

    return df


def load_data():
    csv_path = os.path.join(DATA_DIR, "orders.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=["order_date", "promised_date", "delivered_date"])
    print("No orders.csv found in /data — generating synthetic sample dataset instead.")
    return generate_synthetic_orders()


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="order_id").copy()

    # fix invalid quantities (cannot be <= 0)
    df = df[df["quantity"] > 0]

    # impute missing shipping cost with the median for that warehouse
    df["shipping_cost"] = df.groupby("warehouse_id")["shipping_cost"].transform(
        lambda s: s.fillna(s.median())
    )

    # derived fields used by downstream KPI / modelling scripts
    df["order_cycle_days"] = (df["delivered_date"] - df["order_date"]).dt.total_seconds() / 86400
    df["on_time"] = (df["delivered_date"] <= df["promised_date"]).astype(int)
    df["order_value"] = df["quantity"] * df["unit_cost"]

    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_data()
    print(f"Raw records: {len(raw)}")

    cleaned = clean_orders(raw)
    print(f"Cleaned records: {len(cleaned)}")
    print(cleaned.describe(include="all").T[["count", "mean", "std"]].head(10))

    out_path = os.path.join(DATA_DIR, "orders_cleaned.csv")
    os.makedirs(DATA_DIR, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    print(f"Saved cleaned dataset to {out_path}")
