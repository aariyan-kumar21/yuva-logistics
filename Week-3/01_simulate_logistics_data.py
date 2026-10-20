"""
Script 1: Simulate Logistics Dataset for Advanced EDA & Visualization
Generates 1,800 clean shipment records across the full year of 2025 for
NorthBridge Retail (North, Central, and South fulfillment centers).
Includes realistic seasonal Q4 spikes, distance-based freight costs,
mode-specific transit times, and a Nov-Dec peak-season carrier delay.
"""

import os
import numpy as np
import pandas as pd


def generate_eda_dataset(output_file="data/week3_logistics_eda_data.csv", seed=42):
    np.random.seed(seed)
    n_rows = 1800

    # Generate dates across 2025 with higher order density in Q4 (Oct-Dec peak season)
    months = np.random.choice(
        range(1, 13),
        size=n_rows,
        p=[0.07, 0.06, 0.07, 0.07, 0.08, 0.08, 0.08, 0.08, 0.09, 0.10, 0.11, 0.11]
    )
    days = np.random.randint(1, 29, size=n_rows)
    order_dates = pd.to_datetime([f"2025-{m:02d}-{d:02d}" for m, d in zip(months, days)])
    order_dates = pd.Series(order_dates).sort_values().reset_index(drop=True)

    order_ids = [f"ORD-{20000 + i}" for i in range(n_rows)]

    # Three regional warehouses and four shipping tiers
    regions = np.random.choice(["North", "Central", "South"], size=n_rows, p=[0.35, 0.40, 0.25])
    shipping_modes = np.random.choice(
        ["Standard Class", "Second Class", "First Class", "Same Day"],
        size=n_rows,
        p=[0.55, 0.22, 0.15, 0.08]
    )

    # Product categories with distinct unit values and weights
    categories = np.random.choice(
        ["Electronics", "Home & Office", "Apparel", "Industrial Parts"],
        size=n_rows,
        p=[0.25, 0.30, 0.30, 0.15]
    )

    # Shipment volume (units) and physical package weight (kg)
    shipment_volumes = np.random.randint(1, 25, size=n_rows)
    cat_weight_factor = {"Electronics": 1.8, "Home & Office": 3.2, "Apparel": 0.6, "Industrial Parts": 4.5}
    unit_weights = np.array([cat_weight_factor[c] for c in categories])
    weights_kg = np.round(
        shipment_volumes * unit_weights * np.random.uniform(0.85, 1.15, size=n_rows) + 0.5, 2
    )

    # Route distance in kilometers (regional topology differences)
    region_dist_mean = {"North": 620, "Central": 410, "South": 540}
    base_dist = np.array([region_dist_mean[r] for r in regions])
    distances_km = np.round(np.maximum(50, np.random.normal(base_dist, 180, size=n_rows)), 1)

    # Warehouse processing hours (South hub has an intentional packing bottleneck)
    hub_proc_base = {"North": 8.5, "Central": 7.0, "South": 16.5}
    proc_base = np.array([hub_proc_base[r] for r in regions])
    # Peak season (Nov-Dec) adds extra warehouse congestion
    is_peak = order_dates.dt.month.isin([11, 12]).astype(int)
    warehouse_proc_hours = np.round(
        np.maximum(2.0, np.random.normal(proc_base + is_peak * 4.2, 3.1, size=n_rows)), 1
    )

    # Delivery time in days. Each service tier has a base transit time, a distance penalty
    # (express tiers move faster per km, so distance matters less), the hub processing delay,
    # random carrier noise, and an extra carrier delay during the Nov-Dec peak season.
    mode_base_transit = {"Standard Class": 3.95, "Second Class": 2.65, "First Class": 1.38, "Same Day": 0.35}
    mode_km_per_day = {"Standard Class": 900.0, "Second Class": 1100.0, "First Class": 1800.0, "Same Day": 4000.0}
    mode_noise_scale = {"Standard Class": 0.45, "Second Class": 0.35, "First Class": 0.25, "Same Day": 0.12}
    mode_peak_delay = {"Standard Class": 0.36, "Second Class": 0.28, "First Class": 0.20, "Same Day": 0.096}

    base_transit = np.array([mode_base_transit[m] for m in shipping_modes])
    km_per_day = np.array([mode_km_per_day[m] for m in shipping_modes])
    noise_scale = np.array([mode_noise_scale[m] for m in shipping_modes])
    peak_delay = np.array([mode_peak_delay[m] for m in shipping_modes])
    delivery_time_days = np.round(
        base_transit
        + (distances_km / km_per_day)
        + (warehouse_proc_hours / 24.0)
        + (is_peak.values * peak_delay)
        + np.random.gamma(shape=1.6, scale=noise_scale, size=n_rows),
        2
    )

    # Promised SLA target days per shipping mode
    sla_targets = {"Standard Class": 6.0, "Second Class": 4.5, "First Class": 2.8, "Same Day": 1.2}
    promised_sla_days = np.array([sla_targets[m] for m in shipping_modes])
    late_delivery_flag = (delivery_time_days > promised_sla_days).astype(int)

    # Transportation cost in USD (driven by weight, distance, and expedited shipping multiplier)
    mode_cost_mult = {"Standard Class": 1.0, "Second Class": 1.35, "First Class": 1.85, "Same Day": 2.75}
    cost_mult = np.array([mode_cost_mult[m] for m in shipping_modes])
    transport_cost_usd = np.round(
        (14.0 + (weights_kg * 1.45) + (distances_km * 0.045)) * cost_mult
        + np.random.normal(0, 3.5, size=n_rows),
        2
    )
    transport_cost_usd = np.maximum(8.50, transport_cost_usd)

    # Order value in USD for cost-to-revenue ratio analysis
    cat_price = {"Electronics": 120.0, "Home & Office": 65.0, "Apparel": 35.0, "Industrial Parts": 150.0}
    unit_prices = np.array([cat_price[c] for c in categories])
    order_value_usd = np.round(shipment_volumes * unit_prices * np.random.uniform(0.9, 1.1, size=n_rows), 2)

    df = pd.DataFrame({
        "Order_ID": order_ids,
        "Order_Date": order_dates,
        "Order_Month": order_dates.dt.to_period("M").astype(str),
        "Warehouse_Region": regions,
        "Shipping_Mode": shipping_modes,
        "Product_Category": categories,
        "Shipment_Volume_Units": shipment_volumes,
        "Package_Weight_kg": weights_kg,
        "Distance_km": distances_km,
        "Warehouse_Processing_Hours": warehouse_proc_hours,
        "Delivery_Time_Days": delivery_time_days,
        "Promised_SLA_Days": promised_sla_days,
        "Late_Delivery_Flag": late_delivery_flag,
        "Transport_Cost_USD": transport_cost_usd,
        "Order_Value_USD": order_value_usd
    })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Generated Week 3 EDA dataset: {output_file} ({len(df)} rows, {df.shape[1]} columns).")
    return df


if __name__ == "__main__":
    generate_eda_dataset()
