"""
Script 3: Logistics Data Visualizations
Generates 5 publication-quality charts using Matplotlib and Seaborn to analyze
distributions, cost drivers, regional SLA bottlenecks, seasonal volume surges,
and multi-variable correlations.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def build_all_visualizations(input_file="data/week3_logistics_eda_data.csv", out_dir="charts"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_file)

    # Set clean, professional visual styling
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # -------------------------------------------------------------------------
    # Chart 1: Distribution of Delivery Time & Transport Cost (Histogram + KDE)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.histplot(df["Delivery_Time_Days"], bins=28, kde=True, color="#2b5c8f", ax=axes[0], edgecolor="white")
    axes[0].axvline(df["Delivery_Time_Days"].mean(), color="#d95f02", linestyle="--", linewidth=2,
                    label=f"Mean: {df['Delivery_Time_Days'].mean():.2f} d")
    axes[0].axvline(df["Delivery_Time_Days"].median(), color="#1b9e77", linestyle="-", linewidth=2,
                    label=f"Median: {df['Delivery_Time_Days'].median():.2f} d")
    axes[0].set_title("A) Distribution of Delivery Lead Time (Days)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Delivery Time (Days)")
    axes[0].set_ylabel("Number of Shipments")
    axes[0].legend(frameon=True)

    sns.histplot(df["Transport_Cost_USD"], bins=28, kde=True, color="#386cb0", ax=axes[1], edgecolor="white")
    axes[1].axvline(df["Transport_Cost_USD"].mean(), color="#d95f02", linestyle="--", linewidth=2,
                    label=f"Mean: ${df['Transport_Cost_USD'].mean():.2f}")
    axes[1].axvline(df["Transport_Cost_USD"].median(), color="#1b9e77", linestyle="-", linewidth=2,
                    label=f"Median: ${df['Transport_Cost_USD'].median():.2f}")
    axes[1].set_title("B) Distribution of Transportation Cost (USD)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Transportation Cost (USD)")
    axes[1].set_ylabel("Number of Shipments")
    axes[1].legend(frameon=True)

    plt.tight_layout()
    chart1_path = os.path.join(out_dir, "fig1_distributions_kde.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 2: Delivery Time vs Promised SLA by Shipping Mode & Region (Boxplot)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    mode_order = ["Same Day", "First Class", "Second Class", "Standard Class"]
    sns.boxplot(
        data=df,
        x="Shipping_Mode",
        y="Delivery_Time_Days",
        hue="Warehouse_Region",
        order=mode_order,
        hue_order=["Central", "North", "South"],
        palette=["#4daf4a", "#377eb8", "#e41a1c"],
        ax=ax,
        fliersize=3
    )

    # Plot horizontal SLA reference markers per mode
    sla_map = {"Same Day": 1.2, "First Class": 2.8, "Second Class": 4.5, "Standard Class": 6.0}
    for idx, mode in enumerate(mode_order):
        sla_val = sla_map[mode]
        ax.hlines(y=sla_val, xmin=idx - 0.4, xmax=idx + 0.4, colors="black", linestyles="--", linewidth=1.8)
        if idx == 0:
            ax.plot([], [], color="black", linestyle="--", linewidth=1.8, label="Promised SLA Threshold")

    ax.set_title("Delivery Time Distribution by Shipping Mode & Warehouse Region (vs. SLA Cutoff)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Shipping Mode Service Tier")
    ax.set_ylabel("Actual Delivery Time (Days)")
    ax.legend(title="Origin Hub / SLA", loc="upper left", frameon=True)
    plt.tight_layout()
    chart2_path = os.path.join(out_dir, "fig2_sla_boxplot_by_mode_region.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 3: Freight Cost Drivers - Weight vs Cost by Shipping Mode (Scatter)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.2))
    palette_modes = {"Standard Class": "#2b5c8f", "Second Class": "#1b9e77",
                     "First Class": "#d95f02", "Same Day": "#e7298a"}

    for mode in mode_order:
        sub = df[df["Shipping_Mode"] == mode]
        ax.scatter(
            sub["Package_Weight_kg"],
            sub["Transport_Cost_USD"],
            label=mode,
            color=palette_modes[mode],
            alpha=0.55,
            s=28,
            edgecolors="none"
        )
        # Fit linear trendline per shipping tier
        slope, intercept = np.polyfit(sub["Package_Weight_kg"], sub["Transport_Cost_USD"], 1)
        x_vals = np.array([sub["Package_Weight_kg"].min(), sub["Package_Weight_kg"].max()])
        ax.plot(x_vals, slope * x_vals + intercept, color=palette_modes[mode], linewidth=2.2)

    ax.set_title("Transportation Cost vs. Package Weight Across Shipping Tiers (with OLS Trendlines)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Package Physical Weight (kg)")
    ax.set_ylabel("Transportation Cost (USD)")
    ax.legend(title="Shipping Mode", frameon=True)
    plt.tight_layout()
    chart3_path = os.path.join(out_dir, "fig3_cost_vs_weight_regression.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 4: Monthly Shipment Volume Surge vs Late Delivery Rate (Dual-Axis)
    # -------------------------------------------------------------------------
    monthly = df.groupby("Order_Month").agg(
        Total_Units=("Shipment_Volume_Units", "sum"),
        Order_Count=("Order_ID", "count"),
        Late_Rate_Pct=("Late_Delivery_Flag", lambda x: x.mean() * 100),
        Avg_Proc_Hrs=("Warehouse_Processing_Hours", "mean")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    ax2 = ax1.twinx()

    bars = ax1.bar(monthly["Order_Month"], monthly["Total_Units"], color="#8da0cb", alpha=0.75,
                   label="Monthly Shipment Volume (Units)")
    line = ax2.plot(monthly["Order_Month"], monthly["Late_Rate_Pct"], color="#d95f02", marker="o",
                    linewidth=2.5, markersize=7, label="Late Delivery Rate (%)")

    ax1.set_title("2025 Monthly Shipment Volume vs. Late Delivery Rate (%) — Q4 Peak Bottleneck",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Order Month (2025)")
    ax1.set_ylabel("Total Units Shipped", color="#3b5b92", fontweight="bold")
    ax2.set_ylabel("Late Delivery Rate (%)", color="#d95f02", fontweight="bold")
    ax1.tick_params(axis="x", rotation=30)
    ax2.grid(False)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    chart4_path = os.path.join(out_dir, "fig4_monthly_volume_vs_late_rate.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 5: Multivariate Correlation Matrix & Regional Bottleneck Bar Chart
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    corr_cols = [
        "Shipment_Volume_Units",
        "Package_Weight_kg",
        "Distance_km",
        "Warehouse_Processing_Hours",
        "Delivery_Time_Days",
        "Transport_Cost_USD",
        "Late_Delivery_Flag"
    ]
    short_labels = [
        "Volume (Units)",
        "Weight (kg)",
        "Distance (km)",
        "Hub Proc (Hrs)",
        "Lead Time (d)",
        "Cost (USD)",
        "Late Flag (0/1)"
    ]
    corr_mat = df[corr_cols].corr()
    corr_mat.index = short_labels
    corr_mat.columns = short_labels

    sns.heatmap(
        corr_mat,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.6,
        vmax=1.0,
        cbar=True,
        ax=axes[0],
        annot_kws={"size": 8.5}
    )
    axes[0].set_title("A) Logistics Variables Correlation Heatmap", fontsize=11.5, fontweight="bold")

    # Right panel: Warehouse processing hours vs Late rate by region
    reg_summary = df.groupby("Warehouse_Region").agg(
        Avg_Proc_Hrs=("Warehouse_Processing_Hours", "mean"),
        Late_Rate_Pct=("Late_Delivery_Flag", lambda x: x.mean() * 100)
    ).loc[["Central", "North", "South"]].reset_index()

    x = np.arange(len(reg_summary))
    width = 0.36
    axes[1].bar(x - width / 2, reg_summary["Avg_Proc_Hrs"], width, label="Avg Hub Processing (Hrs)", color="#377eb8")
    axes[1].bar(x + width / 2, reg_summary["Late_Rate_Pct"], width, label="Late Delivery Rate (%)", color="#e41a1c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(reg_summary["Warehouse_Region"])
    axes[1].set_title("B) Hub Processing Time vs. Late Rate by Region", fontsize=11.5, fontweight="bold")
    axes[1].set_ylabel("Hours / Percentage (%)")
    axes[1].legend(frameon=True)

    for i, row in reg_summary.iterrows():
        axes[1].text(i - width / 2, row["Avg_Proc_Hrs"] + 0.6, f"{row['Avg_Proc_Hrs']:.1f}h", ha="center", fontsize=8.5)
        axes[1].text(i + width / 2, row["Late_Rate_Pct"] + 0.6, f"{row['Late_Rate_Pct']:.1f}%", ha="center", fontsize=8.5)

    plt.tight_layout()
    chart5_path = os.path.join(out_dir, "fig5_correlation_and_hub_bottleneck.png")
    plt.savefig(chart5_path, dpi=300)
    plt.close()

    print(f"Successfully generated all 5 charts in '{out_dir}/'.")
    return [chart1_path, chart2_path, chart3_path, chart4_path, chart5_path]


if __name__ == "__main__":
    build_all_visualizations()
