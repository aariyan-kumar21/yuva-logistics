"""
Script 4: Outlier Detection and Treatment
Compares IQR fences against Z-scores on skewed logistics columns.
Flags extreme rows in a separate column for finance/ops review, then caps
the values at the 1.5 * IQR boundaries so they don't break downstream models.
"""

import numpy as np
import pandas as pd


def handle_outliers(
    input_file="data/cleaned_imputed_data.csv",
    output_file="data/outlier_treated_data.csv"
):
    df = pd.read_csv(input_file)
    cols_to_check = ["Order_Item_Weight_kg", "Freight_Cost_USD", "Lead_Time_Days"]

    # Keep an audit flag so we never lose track of which rows had extreme numbers
    df["Outlier_Flag"] = 0

    print("\n--- Outlier Treatment Report (IQR Capping) ---")
    for col in cols_to_check:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower_fence = max(0.0, q1 - 1.5 * iqr)
        upper_fence = q3 + 1.5 * iqr

        # Compare with standard Z-score (> 3 std devs) just to see the difference
        z_vals = np.abs((df[col] - df[col].mean()) / df[col].std())
        z_count = (z_vals > 3.0).sum()

        is_outlier = (df[col] < lower_fence) | (df[col] > upper_fence)
        iqr_count = is_outlier.sum()

        df.loc[is_outlier, "Outlier_Flag"] = 1

        prev_max = df[col].max()
        prev_std = df[col].std()

        # Cap values at the fences instead of deleting rows
        df[col] = np.clip(df[col], lower_fence, upper_fence).round(2)

        print(f"Column: {col}")
        print(f"  Allowed IQR range : {lower_fence:.2f} to {upper_fence:.2f}")
        print(f"  Outliers caught   : {iqr_count} via IQR (vs {z_count} via Z-score)")
        print(f"  Max value change  : {prev_max:.2f} -> {df[col].max():.2f}")
        print(f"  Std Dev change    : {prev_std:.2f} -> {df[col].std():.2f}\n")

    print(f"Total orders flagged in Outlier_Flag: {df['Outlier_Flag'].sum()}")
    df.to_csv(output_file, index=False)
    print(f"Saved outlier-treated dataset to {output_file}.")
    return df


if __name__ == "__main__":
    handle_outliers()
