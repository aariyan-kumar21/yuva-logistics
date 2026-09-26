"""
02_eda.py
---------
Stage 2 of the strategic roadmap: Exploratory Data Analysis & KPI Baselining.

Computes the three core KPIs defined in the strategic report:
  1. On-Time Delivery Rate (OTD)
  2. Average Order Fulfillment Cycle Time
  3. Inventory Turnover Ratio (approximated here via order-value throughput)

and produces summary charts saved to /outputs.

Run: python 02_eda.py
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.dirname(__file__))
from importlib import import_module

# reuse stage-1 cleaning so this script can also run standalone
_dc = import_module("01_data_cleaning")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def compute_kpis(df: pd.DataFrame) -> dict:
    otd_rate = df["on_time"].mean() * 100
    avg_cycle_time = df["order_cycle_days"].mean()

    # Inventory turnover ≈ (annualised units shipped) / (average units on hand).
    # Average inventory on hand is not directly observed here, so it is proxied
    # by a simple average-order-quantity heuristic; replace with actual WMS
    # stock-on-hand figures when available.
    days_span = (df["order_date"].max() - df["order_date"].min()).days or 1
    annualising_factor = 365 / days_span
    total_units_shipped = df["quantity"].sum() * annualising_factor
    avg_inventory_proxy = df.groupby("warehouse_id")["quantity"].mean().mean() * 500
    inventory_turnover = total_units_shipped / avg_inventory_proxy

    return {
        "on_time_delivery_rate_pct": round(otd_rate, 2),
        "avg_order_cycle_time_days": round(avg_cycle_time, 2),
        "inventory_turnover_ratio_annualised": round(inventory_turnover, 2),
    }


def plot_kpis(df: pd.DataFrame):
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    otd_by_wh = df.groupby("warehouse_id")["on_time"].mean() * 100
    otd_by_wh.plot(kind="bar", ax=axes[0], color="#3b6ea5")
    axes[0].set_title("On-Time Delivery Rate by Warehouse")
    axes[0].set_ylabel("OTD %")
    axes[0].set_ylim(0, 100)

    df["order_cycle_days"].plot(kind="hist", bins=30, ax=axes[1], color="#5a9367")
    axes[1].set_title("Order Cycle Time Distribution")
    axes[1].set_xlabel("Days from order to delivery")

    plt.tight_layout()
    fig_path = os.path.join(OUT_DIR, "kpi_baseline.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved KPI chart to {fig_path}")


if __name__ == "__main__":
    raw = _dc.load_data()
    cleaned = _dc.clean_orders(raw)

    kpis = compute_kpis(cleaned)
    print("Baseline KPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    plot_kpis(cleaned)
