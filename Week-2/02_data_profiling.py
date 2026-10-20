"""
Script 2: Initial Data Profiling
Checks the raw CSV for missing values, duplicate order IDs, weird text formatting,
and impossible numbers before we touch anything.
"""

import pandas as pd


def run_data_audit(input_file="data/raw_logistics_data.csv"):
    df = pd.read_csv(input_file)

    print("\n--- Raw Logistics Data Audit ---")
    print(f"Total rows loaded : {len(df)}")
    print(f"Total columns     : {df.shape[1]}")
    print(f"Exact duplicate rows: {df.duplicated().sum()} ({df.duplicated().mean() * 100:.2f}%)")
    print(f"Duplicate Order_IDs : {df['Order_ID'].duplicated().sum()}\n")

    # Check missing values and data types per column
    col_audit = pd.DataFrame({
        "Dtype": df.dtypes,
        "Non_Null": df.notnull().sum(),
        "Missing_Count": df.isnull().sum(),
        "Missing_Pct": (df.isnull().mean() * 100).round(2),
        "Unique_Values": df.nunique()
    })
    print("Column Breakdown:")
    print(col_audit.to_string())
    print()

    # Check distributions on numeric columns
    numeric_cols = ["Order_Item_Quantity", "Order_Item_Weight_kg", "Freight_Cost_USD", "Lead_Time_Days"]
    print("Numeric Summary (including 1st and 99th percentiles):")
    print(df[numeric_cols].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).round(2).to_string())
    print()

    # Look at the messy text variants in Warehouse_Region
    print("Value counts for Warehouse_Region (checking for formatting issues):")
    print(df["Warehouse_Region"].value_counts(dropna=False).to_string())
    print()

    # Check for negative transit days
    neg_days = (df["Lead_Time_Days"] < 0).sum()
    print(f"Rows with impossible negative Lead_Time_Days: {neg_days}\n")
    return col_audit


if __name__ == "__main__":
    run_data_audit()
