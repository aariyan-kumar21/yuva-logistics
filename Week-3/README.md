# Week 3 — Advanced Data Analysis & Visualization in Logistics

This repository contains the Python scripts, dataset, and generated charts for my **Week 3 Logistics Data Analyst Internship** task with Yuvaintern.

Building on our Week 1 KPI roadmap and Week 2 data cleaning pipeline, Week 3 focuses on **Exploratory Data Analysis (EDA) and visual storytelling** for NorthBridge Retail's three fulfillment centers (`North`, `Central`, and `South`). The goal is to uncover what drives freight costs, why certain shipments miss their delivery SLA, and how seasonal volume spikes create warehouse bottlenecks.

## Repository Structure
* **`01_simulate_logistics_data.py`** — Generates a 1,800-shipment dataset (`data/week3_logistics_eda_data.csv`) spanning January–December 2025, capturing shipment volumes, package weights, route distances, warehouse processing hours, transit times, SLA targets, and freight costs.
* **`02_exploratory_data_analysis.py`** — Computes central tendencies (mean, median, mode), dispersion/shape metrics (standard deviation, IQR, skewness, kurtosis), Pearson correlation matrices, and regional KPI summaries.
* **`03_generate_visualizations.py`** — Uses `matplotlib` and `seaborn` to generate 5 high-resolution (300 DPI) analytical charts inside the `charts/` directory:
  1. `fig1_distributions_kde.png` — Histograms & KDE curves for Delivery Lead Time and Transportation Cost.
  2. `fig2_sla_boxplot_by_mode_region.png` — Grouped boxplots comparing actual transit days against SLA cutoffs across shipping tiers and hubs.
  3. `fig3_cost_vs_weight_regression.png` — Multi-tier scatter plot and OLS regression trendlines showing how weight and service speed drive freight spend.
  4. `fig4_monthly_volume_vs_late_rate.png` — Dual-axis time-series chart tracking the Q4 peak volume surge against late-delivery rates.
  5. `fig5_correlation_and_hub_bottleneck.png` — Pearson correlation heatmap paired with a regional warehouse processing bottleneck comparison.
* **`main_pipeline.py`** — Master script that executes all steps sequentially.
* **`requirements.txt`** — Python package dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full pipeline:
   ```bash
   python main_pipeline.py
   ```
