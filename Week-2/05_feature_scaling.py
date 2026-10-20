"""
Script 5: Feature Scaling and Normalization
Adds standardized (Z-score) and normalized (Min-Max 0-1) columns for machine
learning models while keeping the original cleaned columns intact for reporting.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_features(
    input_file="data/outlier_treated_data.csv",
    output_file="data/final_preprocessed_logistics_data.csv"
):
    df = pd.read_csv(input_file)

    # Z-score standardization works best for weight and freight cost (used in regression)
    z_scaler = StandardScaler()
    z_cols = ["Order_Item_Weight_kg", "Freight_Cost_USD"]
    z_vals = z_scaler.fit_transform(df[z_cols])

    df["Weight_kg_ZScore"] = z_vals[:, 0].round(4)
    df["Freight_Cost_ZScore"] = z_vals[:, 1].round(4)

    # Min-Max scaling [0, 1] puts quantity and lead time on a tight bounded scale for K-Means clustering
    mm_scaler = MinMaxScaler(feature_range=(0, 1))
    mm_cols = ["Order_Item_Quantity", "Lead_Time_Days"]
    mm_vals = mm_scaler.fit_transform(df[mm_cols])

    df["Quantity_MinMax"] = mm_vals[:, 0].round(4)
    df["Lead_Time_MinMax"] = mm_vals[:, 1].round(4)

    df.to_csv(output_file, index=False)

    print("\n--- Scaled Features Summary ---")
    new_cols = ["Weight_kg_ZScore", "Freight_Cost_ZScore", "Quantity_MinMax", "Lead_Time_MinMax"]
    print(df[new_cols].describe().loc[["mean", "std", "min", "max"]].round(4).to_string())
    print(f"\nSaved final model-ready dataset to {output_file}.")
    return df


if __name__ == "__main__":
    scale_features()
