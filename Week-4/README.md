# Week 4 — Predictive Modeling and Optimization in Logistics Systems

This repository contains my Python codebase, dataset, and visual outputs for the **Week 4 Logistics Data Analyst Internship** task with Yuvaintern.

Over Weeks 1–3, we established NorthBridge Retail's supply chain KPIs, built a data cleaning pipeline, and ran exploratory analysis that uncovered severe warehouse processing delays at the South hub and high freight costs on expedited tiers. In Week 4, we move from diagnostic analytics to **predictive and prescriptive analytics**:
1. **Predictive Modeling:** Forecast each order's exact `Delivery_Time_Days` prior to warehouse dispatch using `scikit-learn`.
2. **Operations Optimization:** Use those predictions to automatically trigger fast-track dock reallocation or cost-effective carrier tier upgrades whenever an order is predicted to miss its customer SLA.

## Key Results (Dynamically Verified)
* **Best Model (Tuned Random Forest via `GridSearchCV`):** Test RMSE = **0.3469 days**, Test MAE = **0.2804 days**, MAPE = **8.15%**, and $R^2$ = **0.9517**.
* **Prescriptive Optimization Impact (400 Test Orders):** Reduced late deliveries from **114 orders (28.50%)** down to **74 orders (18.50%)**, saving **$1,734.20 (3.51%)** in combined freight and SLA penalty costs.

## Repository Structure
* `01_simulate_modeling_data.py` — Generates `data/week4_logistics_modeling_data.csv` (2,000 shipments).
* `02_train_evaluate_models.py` — Runs 5-Fold Cross-Validation across Ridge Regression, Decision Tree, and Random Forest, tunes hyperparameters via `GridSearchCV`, and exports predictions and feature importances.
* `03_logistics_optimization.py` — Runs the prescriptive dock reallocation and carrier upgrade optimization rules on the 400 test-set shipments.
* `04_generate_model_charts.py` — Generates 4 high-resolution (300 DPI) PNG charts in `charts/`.
* `main_pipeline.py` — Master script that runs all four stages sequentially.
* `requirements.txt` — Python package dependencies.

## Quickstart
```bash
pip install -r requirements.txt
python main_pipeline.py

```
