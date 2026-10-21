"""
Script 3: Prescriptive Logistics Optimization Engine
Uses the Tuned Random Forest model's predicted delivery times on the 400-order
test set to trigger two targeted operational interventions before dispatch:
1. Fast-Track Dock Reallocation (cuts hub processing time by 48% for congested hubs)
2. Selective Carrier Tier Upgrade (upgrades Standard -> Second Class or Second -> First Class
   ONLY when the expected $75 SLA late-penalty savings exceed the incremental freight cost).
"""

import numpy as np
import pandas as pd


def run_operations_optimization(
    input_file="data/test_predictions_for_optimization.csv",
    output_file="data/optimized_test_schedule.csv",
    sla_penalty_usd=75.0
):
    df = pd.read_csv(input_file)

    # Baseline metrics (Before Optimization)
    base_late_count = int(df["Late_Delivery_Flag"].sum())
    base_late_rate_pct = round(df["Late_Delivery_Flag"].mean() * 100, 2)
    base_freight_total = round(df["Freight_Cost_USD"].sum(), 2)
    base_penalty_total = round(base_late_count * sla_penalty_usd, 2)
    base_total_cost = round(base_freight_total + base_penalty_total, 2)

    # Buffer threshold: intervene if predicted delivery time is within 0.15 days of breaching SLA
    at_risk_mask = df["Predicted_Delivery_Days"] > (df["Promised_SLA_Days"] - 0.15)

    opt_days = df["Delivery_Time_Days"].copy()
    opt_freight = df["Freight_Cost_USD"].copy()
    actions = ["No Action Needed"] * len(df)

    # Speed improvement and cost multipliers for carrier tier upgrades
    upgrade_map = {
        "Standard Class": ("Second Class", 1.25, 0.22),   # saves ~1.25 days, +22% freight cost
        "Second Class": ("First Class", 1.10, 0.28),      # saves ~1.10 days, +28% freight cost
    }

    dock_realloc_count = 0
    carrier_upgrade_count = 0

    for idx in df[at_risk_mask].index:
        row = df.loc[idx]
        curr_days = row["Delivery_Time_Days"]
        curr_cost = row["Freight_Cost_USD"]
        sla = row["Promised_SLA_Days"]
        mode = row["Shipping_Mode"]
        proc_hrs = row["Warehouse_Processing_Hours"]

        days_saved_dock = 0.0
        dock_fee = 0.0

        # Lever 1: Fast-Track Dock Reallocation if hub processing > 9.0 hours
        if proc_hrs > 9.0:
            days_saved_dock = round((proc_hrs * 0.48) / 24.0, 2)
            dock_fee = 6.50  # internal priority handling labor cost

        new_days_after_dock = max(0.4, curr_days - days_saved_dock)

        # Check if dock reallocation alone solves the predicted SLA breach
        pred_after_dock = row["Predicted_Delivery_Days"] - days_saved_dock
        if days_saved_dock > 0 and pred_after_dock <= sla:
            opt_days.loc[idx] = round(new_days_after_dock, 2)
            opt_freight.loc[idx] = round(curr_cost + dock_fee, 2)
            actions[df.index.get_loc(idx)] = "Fast-Track Hub Dock"
            dock_realloc_count += 1
            continue

        # Lever 2: Evaluate Carrier Tier Upgrade if still predicted to breach SLA
        if mode in upgrade_map:
            new_mode, transit_days_saved, cost_pct_increase = upgrade_map[mode]
            incremental_freight = round(curr_cost * cost_pct_increase + dock_fee, 2)
            # Only upgrade if incremental cost is strictly less than the $75 SLA penalty
            if incremental_freight < sla_penalty_usd:
                total_saved = days_saved_dock + transit_days_saved
                opt_days.loc[idx] = round(max(0.4, curr_days - total_saved), 2)
                opt_freight.loc[idx] = round(curr_cost + incremental_freight, 2)
                actions[df.index.get_loc(idx)] = f"Upgrade to {new_mode} + Priority Dock"
                carrier_upgrade_count += 1
        elif days_saved_dock > 0:
            # Apply dock fast-tracking for First Class / Same Day orders
            opt_days.loc[idx] = round(new_days_after_dock, 2)
            opt_freight.loc[idx] = round(curr_cost + dock_fee, 2)
            actions[df.index.get_loc(idx)] = "Fast-Track Hub Dock"
            dock_realloc_count += 1

    df["Optimization_Action"] = actions
    df["Optimized_Delivery_Days"] = opt_days
    df["Optimized_Late_Flag"] = (df["Optimized_Delivery_Days"] > df["Promised_SLA_Days"]).astype(int)
    df["Optimized_Freight_USD"] = opt_freight

    opt_late_count = int(df["Optimized_Late_Flag"].sum())
    opt_late_rate_pct = round(df["Optimized_Late_Flag"].mean() * 100, 2)
    opt_freight_total = round(df["Optimized_Freight_USD"].sum(), 2)
    opt_penalty_total = round(opt_late_count * sla_penalty_usd, 2)
    opt_total_cost = round(opt_freight_total + opt_penalty_total, 2)

    net_savings_usd = round(base_total_cost - opt_total_cost, 2)
    cost_reduction_pct = round((net_savings_usd / base_total_cost) * 100, 2)

    df.to_csv(output_file, index=False)

    summary_dict = {
        "Test_Orders": len(df),
        "At_Risk_Identified": int(at_risk_mask.sum()),
        "Dock_Reallocations": dock_realloc_count,
        "Carrier_Upgrades": carrier_upgrade_count,
        "Base_Late_Count": base_late_count,
        "Opt_Late_Count": opt_late_count,
        "Base_Late_Rate_Pct": base_late_rate_pct,
        "Opt_Late_Rate_Pct": opt_late_rate_pct,
        "Base_Freight_USD": base_freight_total,
        "Opt_Freight_USD": opt_freight_total,
        "Base_Penalty_USD": base_penalty_total,
        "Opt_Penalty_USD": opt_penalty_total,
        "Base_Total_Cost_USD": base_total_cost,
        "Opt_Total_Cost_USD": opt_total_cost,
        "Net_Savings_USD": net_savings_usd,
        "Cost_Reduction_Pct": cost_reduction_pct
    }

    # Regional breakdown before vs after
    reg_opt = df.groupby("Warehouse_Region").agg(
        Orders=("Order_ID", "count"),
        Base_Late_Pct=("Late_Delivery_Flag", lambda s: round(s.mean() * 100, 2)),
        Opt_Late_Pct=("Optimized_Late_Flag", lambda s: round(s.mean() * 100, 2)),
        Base_Avg_Days=("Delivery_Time_Days", lambda s: round(s.mean(), 2)),
        Opt_Avg_Days=("Optimized_Delivery_Days", lambda s: round(s.mean(), 2))
    ).reset_index()
    reg_opt.to_csv("data/regional_optimization_impact.csv", index=False)

    print("\n--- Prescriptive Optimization Summary (400 Test Shipments) ---")
    for k, v in summary_dict.items():
        print(f"  {k:22s}: {v}")
    print("\n--- Regional SLA Improvement ---")
    print(reg_opt.to_string(index=False))

    return summary_dict, reg_opt


if __name__ == "__main__":
    run_operations_optimization()
