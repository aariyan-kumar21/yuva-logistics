"""
Script 2: Exploratory Data Analysis (EDA)
Calculates central tendencies (mean, median, mode), dispersion and shape
metrics (std dev, IQR, skewness, kurtosis), Pearson correlations,
and regional/mode KPI breakdowns.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def perform_eda(input_file="data/week3_logistics_eda_data.csv"):
    df = pd.read_csv(input_file)

    num_cols = [
        "Shipment_Volume_Units",
        "Package_Weight_kg",
        "Distance_km",
        "Warehouse_Processing_Hours",
        "Delivery_Time_Days",
        "Transport_Cost_USD",
        "Order_Value_USD"
    ]

    print("\n--- 1. Central Tendency & Distribution Shape Metrics ---")
    stats_rows = []
    for col in num_cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        stats_rows.append({
            "Variable": col,
            "Mean": round(s.mean(), 2),
            "Median": round(s.median(), 2),
            "Mode": round(s.mode().iloc[0], 2),
            "Std_Dev": round(s.std(), 2),
            "IQR": round(q3 - q1, 2),
            "Min": round(s.min(), 2),
            "Max": round(s.max(), 2),
            "Skewness": round(skew(s), 2),
            "Kurtosis": round(kurtosis(s), 2)
        })
    stats_df = pd.DataFrame(stats_rows)
    print(stats_df.to_string(index=False))

    print("\n--- 2. Pearson Correlation Matrix ---")
    corr_matrix = df[num_cols + ["Late_Delivery_Flag"]].corr(method="pearson").round(3)
    print(corr_matrix.to_string())

    print("\n--- 3. Performance Breakdown by Warehouse Region ---")
    region_kpi = df.groupby("Warehouse_Region").agg(
        Total_Orders=("Order_ID", "count"),
        Avg_Processing_Hrs=("Warehouse_Processing_Hours", "mean"),
        Avg_Delivery_Days=("Delivery_Time_Days", "mean"),
        Avg_Transport_Cost=("Transport_Cost_USD", "mean"),
        Late_Delivery_Rate_Pct=("Late_Delivery_Flag", lambda x: round(x.mean() * 100, 2))
    ).round(2)
    print(region_kpi.to_string())

    print("\n--- 4. Performance Breakdown by Shipping Mode ---")
    mode_kpi = df.groupby("Shipping_Mode").agg(
        Order_Share=("Order_ID", "count"),
        Avg_Delivery_Days=("Delivery_Time_Days", "mean"),
        Promised_SLA_Days=("Promised_SLA_Days", "first"),
        Avg_Transport_Cost=("Transport_Cost_USD", "mean"),
        Late_Delivery_Rate_Pct=("Late_Delivery_Flag", lambda x: round(x.mean() * 100, 2))
    ).round(2)
    print(mode_kpi.to_string())

    return stats_df, corr_matrix, region_kpi, mode_kpi


if __name__ == "__main__":
    perform_eda()
