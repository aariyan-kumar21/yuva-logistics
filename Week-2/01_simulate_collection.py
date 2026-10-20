"""
Script 1: Simulate Logistics Data Collection
Builds a synthetic dataset modeled after the DataCo Smart Supply Chain dataset.
Includes realistic operational messiness (duplicates, missing scale reads,
billing typos, and timezone glitches) so we can test our cleaning pipeline.
"""

import os
import numpy as np
import pandas as pd


def create_raw_dataset(output_file="data/raw_logistics_data.csv", seed=42):
    np.random.seed(seed)
    n_rows = 1500

    # Generate basic order IDs and hourly timestamps starting Jan 2026
    order_ids = [f"ORD-{10000 + i}" for i in range(n_rows)]
    order_dates = pd.date_range(start="2026-01-01", periods=n_rows, freq="h")

    # Simulate messy regional strings coming from different warehouse terminals
    region_choices = [
        "North", " north ", "NORTH",
        "Central", "central ", "CENTRAL",
        "South", " south", "SOUTH"
    ]
    region_probs = [0.25, 0.05, 0.05, 0.30, 0.03, 0.02, 0.25, 0.03, 0.02]
    regions = np.random.choice(region_choices, size=n_rows, p=region_probs)

    shipping_tiers = np.random.choice(
        ["Standard Class", "Second Class", "First Class", "Same Day"],
        size=n_rows,
        p=[0.60, 0.20, 0.15, 0.05]
    ).astype(object)

    # Order quantities between 1 and 14 units
    quantities = np.random.randint(1, 15, size=n_rows)

    # Package weight scales with item quantity plus a bit of random packaging weight
    raw_weights = quantities * np.random.uniform(1.2, 3.5, size=n_rows) + np.random.normal(0, 0.8, size=n_rows)
    weights = np.round(np.maximum(raw_weights, 0.5), 2)

    # Calculate freight cost based on weight and how fast the customer wants it shipped
    rate_multipliers = {
        "Standard Class": 1.0,
        "Second Class": 1.35,
        "First Class": 1.75,
        "Same Day": 2.50
    }
    tier_mult = np.array([rate_multipliers[m] for m in shipping_tiers])
    raw_costs = (12.0 + weights * 1.85) * tier_mult + np.random.normal(0, 4.0, size=n_rows)
    freight_costs = np.round(np.maximum(raw_costs, 5.0), 2)

    # Transit lead time (in days) based on shipping tier
    avg_transit = {
        "Standard Class": 5.5,
        "Second Class": 4.0,
        "First Class": 2.2,
        "Same Day": 0.8
    }
    base_days = np.array([avg_transit[m] for m in shipping_tiers])
    lead_times = np.round(base_days + np.random.gamma(shape=2.0, scale=0.8, size=n_rows), 1)

    shipping_dates = order_dates + pd.to_timedelta(lead_times, unit="D")

    # Flag orders that missed their promised delivery window
    sla_cutoffs = {
        "Standard Class": 6.0,
        "Second Class": 4.5,
        "First Class": 2.5,
        "Same Day": 1.0
    }
    max_allowed_days = np.array([sla_cutoffs[m] for m in shipping_tiers])
    late_flag = (lead_times > max_allowed_days).astype(int)

    df = pd.DataFrame({
        "Order_ID": order_ids,
        "Order_Date": order_dates,
        "Shipping_Date": shipping_dates,
        "Warehouse_Region": regions,
        "Shipping_Mode": shipping_tiers,
        "Order_Item_Quantity": quantities,
        "Order_Item_Weight_kg": weights,
        "Freight_Cost_USD": freight_costs,
        "Lead_Time_Days": lead_times,
        "Late_Delivery_Risk": late_flag
    })

    # --- Inject realistic data quality problems ---

    # 1. Missing values (skipped weight scans, delayed carrier invoices, unmapped mode codes)
    missing_weight_rows = np.random.choice(df.index, size=120, replace=False)
    missing_cost_rows = np.random.choice(df.index, size=90, replace=False)
    missing_mode_rows = np.random.choice(df.index, size=45, replace=False)

    df.loc[missing_weight_rows, "Order_Item_Weight_kg"] = np.nan
    df.loc[missing_cost_rows, "Freight_Cost_USD"] = np.nan
    df.loc[missing_mode_rows, "Shipping_Mode"] = np.nan

    # 2. Negative lead times caused by server timezone mismatches
    bad_time_rows = np.random.choice(df.index, size=18, replace=False)
    df.loc[bad_time_rows, "Lead_Time_Days"] = np.round(np.random.uniform(-4.5, -0.5, size=18), 1)

    # 3. Extreme outliers (manual decimal typos on freight bills and heavy bulk orders)
    valid_cost_idx = df.index.difference(missing_cost_rows)
    cost_spike_rows = np.random.choice(valid_cost_idx, size=12, replace=False)
    df.loc[cost_spike_rows, "Freight_Cost_USD"] = np.round(np.random.uniform(1200.0, 4850.0, size=12), 2)

    valid_weight_idx = df.index.difference(missing_weight_rows)
    weight_spike_rows = np.random.choice(valid_weight_idx, size=10, replace=False)
    df.loc[weight_spike_rows, "Order_Item_Weight_kg"] = np.round(np.random.uniform(350.0, 780.0, size=10), 2)

    # 4. Duplicate records from system retry glitches
    dup_sample = df.sample(n=35, random_state=seed)
    df_messy = pd.concat([df, dup_sample], ignore_index=True)
    df_messy = df_messy.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_messy.to_csv(output_file, index=False)
    print(f"Saved raw dataset to {output_file} ({df_messy.shape[0]} rows, {df_messy.shape[1]} columns).")
    return df_messy


if __name__ == "__main__":
    create_raw_dataset()
