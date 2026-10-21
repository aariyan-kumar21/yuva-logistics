"""
Script 1: Simulate Logistics Dataset for Predictive Modeling & Optimization
Generates 2,000 shipment records for NorthBridge Retail across 3 fulfillment
centers (North, Central, South) with realistic non-linear interactions between
distance, warehouse queue congestion, weather/traffic indices, and transit time.
"""

import os
import numpy as np
import pandas as pd


def generate_modeling_dataset(output_file="data/week4_logistics_modeling_data.csv", seed=42):
    np.random.seed(seed)
    n_rows = 2000

    order_ids = [f"ORD-{30000 + i}" for i in range(n_rows)]

    # Regional warehouses and carrier service tiers
    regions = np.random.choice(["North", "Central", "South"], size=n_rows, p=[0.35, 0.40, 0.25])
    shipping_modes = np.random.choice(
        ["Standard Class", "Second Class", "First Class", "Same Day"],
        size=n_rows,
        p=[0.52, 0.23, 0.16, 0.09]
    )

    # Order characteristics
    order_quantities = np.random.randint(1, 25, size=n_rows)
    package_weights_kg = np.round(
        order_quantities * np.random.uniform(0.9, 3.2, size=n_rows) + np.random.uniform(0.5, 2.5, size=n_rows), 2
    )

    # Route distance in kilometers
    reg_dist = {"North": 610, "Central": 430, "South": 550}
    base_d = np.array([reg_dist[r] for r in regions])
    distance_km = np.round(np.maximum(45.0, np.random.normal(base_d, 175.0, size=n_rows)), 1)

    # Operational conditions at dispatch:
    # 1. Warehouse Queue Utilization % (South hub runs hotter/more congested)
    reg_util = {"North": 68.0, "Central": 62.0, "South": 84.0}
    base_u = np.array([reg_util[r] for r in regions])
    hub_utilization_pct = np.round(np.clip(np.random.normal(base_u, 11.5, size=n_rows), 30.0, 99.5), 1)

    # 2. Warehouse Processing Hours (non-linear spike when utilization exceeds 80%)
    congestion_penalty = np.where(hub_utilization_pct > 80.0, (hub_utilization_pct - 80.0) * 0.48, 0.0)
    warehouse_proc_hours = np.round(
        np.maximum(2.0, 5.5 + (hub_utilization_pct * 0.08) + congestion_penalty + np.random.normal(0, 1.4, size=n_rows)), 2
    )

    # 3. Route Traffic & Weather Disruption Index (1.0 = clear, 10.0 = severe storm/gridlock)
    disruption_index = np.round(np.random.beta(a=2.0, b=5.0, size=n_rows) * 9.0 + 1.0, 2)

    # 4. Peak Season Flag (1 = Q4 holiday peak, 0 = regular season)
    is_peak_season = np.random.choice([0, 1], size=n_rows, p=[0.78, 0.22])

    # Target Variable: Actual Delivery_Time_Days (includes non-linear effects)
    mode_base_days = {"Standard Class": 3.4, "Second Class": 2.2, "First Class": 1.1, "Same Day": 0.35}
    mode_speed_km_per_day = {"Standard Class": 390.0, "Second Class": 460.0, "First Class": 620.0, "Same Day": 950.0}

    base_transit = np.array([mode_base_days[m] for m in shipping_modes])
    speed_factor = np.array([mode_speed_km_per_day[m] for m in shipping_modes])

    # Non-linear interaction: long distances during high traffic disruption take extra time
    weather_distance_delay = (distance_km / 500.0) * np.maximum(0, (disruption_index - 4.0) * 0.28)

    delivery_time_days = np.round(
        base_transit
        + (distance_km / speed_factor)
        + (warehouse_proc_hours / 24.0)
        + weather_distance_delay
        + (is_peak_season * 0.55)
        + np.random.normal(0, 0.32, size=n_rows),
        2
    )
    delivery_time_days = np.maximum(0.4, delivery_time_days)

    # Contractual SLA Cutoff Days & Late Flag
    sla_cutoffs = {"Standard Class": 6.0, "Second Class": 4.5, "First Class": 2.8, "Same Day": 1.5}
    promised_sla_days = np.array([sla_cutoffs[m] for m in shipping_modes])
    late_delivery_flag = (delivery_time_days > promised_sla_days).astype(int)

    # Baseline Carrier Freight Cost (USD)
    mode_mult = {"Standard Class": 1.0, "Second Class": 1.32, "First Class": 1.80, "Same Day": 2.65}
    mult_arr = np.array([mode_mult[m] for m in shipping_modes])
    freight_cost_usd = np.round(
        (12.50 + (package_weights_kg * 1.35) + (distance_km * 0.042)) * mult_arr, 2
    )

    df = pd.DataFrame({
        "Order_ID": order_ids,
        "Warehouse_Region": regions,
        "Shipping_Mode": shipping_modes,
        "Order_Item_Quantity": order_quantities,
        "Package_Weight_kg": package_weights_kg,
        "Distance_km": distance_km,
        "Hub_Utilization_Pct": hub_utilization_pct,
        "Warehouse_Processing_Hours": warehouse_proc_hours,
        "Disruption_Index": disruption_index,
        "Is_Peak_Season": is_peak_season,
        "Promised_SLA_Days": promised_sla_days,
        "Freight_Cost_USD": freight_cost_usd,
        "Delivery_Time_Days": delivery_time_days,
        "Late_Delivery_Flag": late_delivery_flag
    })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved Week 4 modeling dataset to {output_file} ({len(df)} rows, {df.shape[1]} columns).")
    return df


if __name__ == "__main__":
    generate_modeling_dataset()
