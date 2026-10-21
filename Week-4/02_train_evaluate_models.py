"""
Script 2: Predictive Modeling, Cross-Validation, and Hyperparameter Tuning
Prepares features (One-Hot Encoding + Scaling), splits data 80/20, compares
Ridge Regression, Decision Tree, and Random Forest via 5-Fold Cross-Validation,
and tunes the Random Forest using GridSearchCV.
"""

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def train_and_evaluate_models(input_file="data/week4_logistics_modeling_data.csv"):
    df = pd.read_csv(input_file)

    feature_cols = [
        "Warehouse_Region",
        "Shipping_Mode",
        "Order_Item_Quantity",
        "Package_Weight_kg",
        "Distance_km",
        "Hub_Utilization_Pct",
        "Warehouse_Processing_Hours",
        "Disruption_Index",
        "Is_Peak_Season"
    ]
    target_col = "Delivery_Time_Days"

    X = df[feature_cols]
    y = df[target_col]

    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.20, random_state=42
    )

    cat_cols = ["Warehouse_Region", "Shipping_Mode"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
        ]
    )

    # Define candidate models
    models = {
        "Ridge Linear Regression": Ridge(alpha=1.0, random_state=42),
        "Decision Tree (Baseline)": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest (Baseline)": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    trained_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("reg", model)])
        cv_neg_mse = cross_val_score(pipe, X_train, y_train, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
        cv_rmse = np.sqrt(-cv_neg_mse)

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        test_rmse = np.sqrt(mean_squared_error(y_test, preds))
        test_mae = mean_absolute_error(y_test, preds)
        test_mape = mean_absolute_percentage_error(y_test, preds) * 100
        test_r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "CV_RMSE_Mean": round(cv_rmse.mean(), 4),
            "CV_RMSE_Std": round(cv_rmse.std(), 4),
            "Test_RMSE": round(test_rmse, 4),
            "Test_MAE": round(test_mae, 4),
            "Test_MAPE_Pct": round(test_mape, 2),
            "Test_R2": round(test_r2, 4)
        })
        trained_pipelines[name] = pipe

    # Hyperparameter Tuning on Random Forest using GridSearchCV
    rf_pipe = Pipeline([("prep", preprocessor), ("reg", RandomForestRegressor(random_state=42, n_jobs=-1))])
    param_grid = {
        "reg__n_estimators": [100, 200],
        "reg__max_depth": [8, 12, None],
        "reg__min_samples_split": [2, 5],
        "reg__min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        rf_pipe, param_grid=param_grid, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    best_cv_rmse = np.sqrt(-grid_search.best_score_)
    best_preds = best_rf.predict(X_test)

    # Also compute std of CV RMSE for best model
    best_cv_scores = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring="neg_mean_squared_error", n_jobs=-1)
    best_cv_std = np.sqrt(-best_cv_scores).std()

    results.append({
        "Model": "Random Forest (Tuned GridSearchCV)",
        "CV_RMSE_Mean": round(best_cv_rmse, 4),
        "CV_RMSE_Std": round(best_cv_std, 4),
        "Test_RMSE": round(np.sqrt(mean_squared_error(y_test, best_preds)), 4),
        "Test_MAE": round(mean_absolute_error(y_test, best_preds), 4),
        "Test_MAPE_Pct": round(mean_absolute_percentage_error(y_test, best_preds) * 100, 2),
        "Test_R2": round(r2_score(y_test, best_preds), 4)
    })
    trained_pipelines["Random Forest (Tuned GridSearchCV)"] = best_rf

    metrics_df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    metrics_df.to_csv("data/model_evaluation_metrics.csv", index=False)

    # Extract Feature Importances from Tuned Random Forest
    ohe_Setting = best_rf.named_steps["prep"].named_transformers_["cat"]
    cat_feature_names = list(ohe_Setting.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    importances = best_rf.named_steps["reg"].feature_importances_

    feat_imp_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance_Pct": np.round(importances * 100, 2)
    }).sort_values(by="Importance_Pct", ascending=False).reset_index(drop=True)
    feat_imp_df.to_csv("data/feature_importances.csv", index=False)

    # Save Test Set Predictions for Optimization & Residual Diagnostics
    df_test_out = df_test.copy()
    df_test_out["Predicted_Delivery_Days"] = np.round(best_preds, 2)
    df_test_out["Residual_Days"] = np.round(df_test_out["Delivery_Time_Days"] - best_preds, 2)
    df_test_out["Predicted_Late_Flag"] = (df_test_out["Predicted_Delivery_Days"] > df_test_out["Promised_SLA_Days"]).astype(int)
    df_test_out.to_csv("data/test_predictions_for_optimization.csv", index=False)

    print("\n--- Model Performance Summary (5-Fold CV & Test Set) ---")
    print(metrics_df.to_string(index=False))
    print("\nBest Hyperparameters:", grid_search.best_params_)
    print("\n--- Top Feature Importances (Tuned Random Forest) ---")
    print(feat_imp_df.to_string(index=False))

    return metrics_df, feat_imp_df, df_test_out, grid_search.best_params_


if __name__ == "__main__":
    train_and_evaluate_models()
