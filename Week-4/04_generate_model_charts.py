"""
Script 4: Generate Predictive Modeling & Optimization Charts
Produces 4 high-resolution (300 DPI) visualizations in charts/ illustrating
model evaluation metrics, actual vs. predicted residuals, feature importances,
and before/after prescriptive optimization results.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_all_charts(out_dir="charts"):
    os.makedirs(out_dir, exist_ok=True)
    metrics_df = pd.read_csv("data/model_evaluation_metrics.csv")
    feat_df = pd.read_csv("data/feature_importances.csv")
    pred_df = pd.read_csv("data/optimized_test_schedule.csv")
    reg_opt = pd.read_csv("data/regional_optimization_impact.csv")

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "sans-serif"

    # -------------------------------------------------------------------------
    # Chart 1: Model Performance Comparison (Test RMSE, MAE, and R-Squared)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    short_names = ["Ridge\nRegression", "Decision Tree\n(Baseline)", "Random Forest\n(Baseline)", "Random Forest\n(Tuned)"]
    x = np.arange(len(short_names))
    width = 0.36

    axes[0].bar(x - width / 2, metrics_df["Test_RMSE"], width, label="Test RMSE (Days)", color="#2b5c8f")
    axes[0].bar(x + width / 2, metrics_df["Test_MAE"], width, label="Test MAE (Days)", color="#38a3a5")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, fontsize=9.5)
    axes[0].set_title("A) Prediction Error by Model (Lower is Better)", fontsize=11.5, fontweight="bold")
    axes[0].set_ylabel("Error in Delivery Days")
    axes[0].legend(frameon=True)
    for i, (rmse_v, mae_v) in enumerate(zip(metrics_df["Test_RMSE"], metrics_df["Test_MAE"])):
        axes[0].text(i - width / 2, rmse_v + 0.015, f"{rmse_v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
        axes[0].text(i + width / 2, mae_v + 0.015, f"{mae_v:.3f}", ha="center", fontsize=8.5)

    bars_r2 = axes[1].bar(short_names, metrics_df["Test_R2"], color=["#8da0cb", "#fc8d62", "#66c2a5", "#1b9e77"], width=0.52)
    axes[1].set_ylim(0.70, 1.0)
    axes[1].set_title("B) Explained Variance R² Score on Test Set (Higher is Better)", fontsize=11.5, fontweight="bold")
    axes[1].set_ylabel("R-Squared (R²)")
    for bar, r2_v in zip(bars_r2, metrics_df["Test_R2"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, r2_v + 0.006, f"{r2_v:.4f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    c1 = os.path.join(out_dir, "fig1_model_comparison_metrics.png")
    plt.savefig(c1, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 2: Actual vs Predicted Delivery Time & Residual Error Distribution
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    axes[0].scatter(pred_df["Delivery_Time_Days"], pred_df["Predicted_Delivery_Days"],
                    alpha=0.55, color="#2b5c8f", edgecolors="none", s=28)
    min_v = min(pred_df["Delivery_Time_Days"].min(), pred_df["Predicted_Delivery_Days"].min())
    max_v = max(pred_df["Delivery_Time_Days"].max(), pred_df["Predicted_Delivery_Days"].max())
    axes[0].plot([min_v, max_v], [min_v, max_v], color="#d95f02", linestyle="--", linewidth=2, label="Perfect Fit (y = x)")
    axes[0].set_title("A) Actual vs. Predicted Delivery Time (Tuned RF)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Actual Delivery Time (Days)")
    axes[0].set_ylabel("Predicted Delivery Time (Days)")
    axes[0].legend(frameon=True)

    sns.histplot(pred_df["Residual_Days"], bins=26, kde=True, color="#1b9e77", ax=axes[1], edgecolor="white")
    axes[1].axvline(0.0, color="#d95f02", linestyle="--", linewidth=2, label="Zero Error Line")
    axes[1].set_title("B) Residual Error Distribution (Actual - Predicted)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Residual Error (Days)")
    axes[1].set_ylabel("Shipment Count")
    axes[1].legend(frameon=True)

    plt.tight_layout()
    c2 = os.path.join(out_dir, "fig2_actual_vs_predicted_residuals.png")
    plt.savefig(c2, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 3: Random Forest Feature Importance Ranking
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    feat_sorted = feat_df.sort_values(by="Importance_Pct", ascending=True)
    bars = ax.barh(feat_sorted["Feature"], feat_sorted["Importance_Pct"], color="#2b5c8f", edgecolor="white")
    ax.set_title("Tuned Random Forest Feature Importance (% Contribution to Delivery Time Prediction)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Relative Importance (%)")
    ax.set_xlim(0, feat_sorted["Importance_Pct"].max() * 1.15)
    for bar, imp in zip(bars, feat_sorted["Importance_Pct"]):
        ax.text(imp + 0.6, bar.get_y() + bar.get_height() / 2, f"{imp:.2f}%", va="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    c3 = os.path.join(out_dir, "fig3_feature_importance_ranking.png")
    plt.savefig(c3, dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Chart 4: Prescriptive Optimization Impact (Regional SLA & Total Cost)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))

    rx = np.arange(len(reg_opt))
    w = 0.35
    axes[0].bar(rx - w / 2, reg_opt["Base_Late_Pct"], w, label="Baseline Late Rate (%)", color="#e41a1c")
    axes[0].bar(rx + w / 2, reg_opt["Opt_Late_Pct"], w, label="Optimized Late Rate (%)", color="#4daf4a")
    axes[0].set_xticks(rx)
    axes[0].set_xticklabels(reg_opt["Warehouse_Region"])
    axes[0].set_title("A) Regional Late-Delivery Rate: Before vs. After Optimization", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Late Delivery Rate (%)")
    axes[0].legend(frameon=True)
    for i, (b_val, o_val) in enumerate(zip(reg_opt["Base_Late_Pct"], reg_opt["Opt_Late_Pct"])):
        axes[0].text(i - w / 2, b_val + 0.6, f"{b_val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        axes[0].text(i + w / 2, o_val + 0.6, f"{o_val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

    # Right panel: Total Cost Breakdown (Freight vs SLA Penalties)
    base_freight = pred_df["Freight_Cost_USD"].sum()
    base_pen = pred_df["Late_Delivery_Flag"].sum() * 75.0
    opt_freight = pred_df["Optimized_Freight_USD"].sum()
    opt_pen = pred_df["Optimized_Late_Flag"].sum() * 75.0

    categories = ["Baseline Operations\n(Unoptimized)", "Prescriptive Policy\n(Model-Guided)"]
    freight_vals = [base_freight, opt_freight]
    pen_vals = [base_pen, opt_pen]

    axes[1].bar(categories, freight_vals, width=0.45, label="Carrier Freight Spend ($)", color="#377eb8")
    axes[1].bar(categories, pen_vals, width=0.45, bottom=freight_vals, label="SLA Late Penalties ($75/order)", color="#ff7f00")
    axes[1].set_title("B) Total Network Cost Breakdown (400 Test Shipments)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Total Cost (USD)")
    axes[1].legend(frameon=True, loc="upper right")

    for i, (fv, pv) in enumerate(zip(freight_vals, pen_vals)):
        tot = fv + pv
        axes[1].text(i, tot + 400, f"Total: ${tot:,.0f}", ha="center", fontsize=9.5, fontweight="bold")

    axes[1].set_ylim(0, max(base_freight + base_pen, opt_freight + opt_pen) * 1.15)

    plt.tight_layout()
    c4 = os.path.join(out_dir, "fig4_optimization_impact.png")
    plt.savefig(c4, dpi=300)
    plt.close()

    print(f"Saved all 4 Week 4 visualizations in '{out_dir}/'.")
    return [c1, c2, c3, c4]


if __name__ == "__main__":
    generate_all_charts()
