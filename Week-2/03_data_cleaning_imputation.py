"""
Script 3: Data Cleaning and Missing Value Imputation
Fixes text casing, drops duplicate orders, turns negative lead times into NaNs,
and fills missing values using group-based medians and modes.
"""

import numpy as np
import pandas as pd


def clean_logistics_data(
    input_file="data/raw_logistics_data.csv",
    output_file="data/cleaned_imputed_data.csv"
):
    df = pd.read_csv(input_file)
    start_count = len(df)

    # 1. Clean up whitespace and inconsistent capitalization in text columns
    df["Warehouse_Region"] = df["Warehouse_Region"].astype(str).str.strip().str.title()
    df["Shipping_Mode"] = df["Shipping_Mode"].replace("nan", np.nan)
    df["Shipping_Mode"] = df["Shipping_Mode"].str.strip().str.title()

    # 2. Drop duplicate orders (keep the first logged entry)
    df = df.drop_duplicates(subset=["Order_ID"], keep="first").reset_index(drop=True)
    removed_dups = start_count - len(df)

    # 3. Convert date strings to proper datetime objects
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    df["Shipping_Date"] = pd.to_datetime(df["Shipping_Date"], errors="coerce")

    # 4. Negative lead times are impossible. Set them to NaN so we can impute them properly
    # without throwing away the rest of the order's financial data.
    bad_lead_mask = df["Lead_Time_Days"] < 0
    bad_lead_count = bad_lead_mask.sum()
    df.loc[bad_lead_mask, "Lead_Time_Days"] = np.nan

    # 5. Impute missing values
    # Fill missing Shipping_Mode with the most common tier (Standard Class)
    most_common_mode = df["Shipping_Mode"].mode()[0]
    df["Shipping_Mode"] = df["Shipping_Mode"].fillna(most_common_mode)

    # Weight depends heavily on how many items were ordered, so we use median weight per quantity
    df["Order_Item_Weight_kg"] = df.groupby("Order_Item_Quantity")["Order_Item_Weight_kg"].transform(
        lambda s: s.fillna(s.median())
    )
    # Quick fallback in case a specific quantity bucket was completely empty
    df["Order_Item_Weight_kg"] = df["Order_Item_Weight_kg"].fillna(df["Order_Item_Weight_kg"].median())

    # Freight cost and transit time depend on both warehouse region and shipping speed
    df["Freight_Cost_USD"] = df.groupby(["Warehouse_Region", "Shipping_Mode"])["Freight_Cost_USD"].transform(
        lambda s: s.fillna(s.median())
    )
    df["Lead_Time_Days"] = df.groupby(["Warehouse_Region", "Shipping_Mode"])["Lead_Time_Days"].transform(
        lambda s: s.fillna(s.median())
    )

    # Fix Shipping_Date for the rows where we just repaired negative lead times
    df.loc[bad_lead_mask, "Shipping_Date"] = (
        df.loc[bad_lead_mask, "Order_Date"] + pd.to_timedelta(df.loc[bad_lead_mask, "Lead_Time_Days"], unit="D")
    )

    df.to_csv(output_file, index=False)
    print(f"Cleaning finished: removed {removed_dups} duplicates, fixed {bad_lead_count} negative lead times.")
    print(f"Remaining nulls in dataset: {df.isnull().sum().sum()}")
    print(f"Saved cleaned file to {output_file}.")
    return df


if __name__ == "__main__":
    clean_logistics_data()
