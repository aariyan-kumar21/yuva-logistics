"""
06_delay_prediction_model.py
------------------------------
Stage 3 (continued): Predictive Modelling — Late-Delivery Risk Classification.

Trains a classifier to flag orders at high risk of missing their promised
delivery date *before* they ship, so operations can proactively expedite
them (upgrade carrier service, reprioritize picking) — directly targeting
the On-Time Delivery Rate KPI.

Model: Logistic Regression baseline vs. Random Forest Classifier.

Run: python 06_delay_prediction_model.py
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

sys.path.append(os.path.dirname(__file__))
from importlib import import_module
_dc = import_module("01_data_cleaning")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["order_dow"] = feat["order_date"].dt.dayofweek
    feat["order_month"] = feat["order_date"].dt.month
    feat["is_late"] = 1 - feat["on_time"]  # target: 1 = delivered late

    feat = pd.get_dummies(feat, columns=["warehouse_id", "dest_zone"], drop_first=True)
    return feat


def train_and_evaluate(feat: pd.DataFrame):
    drop_cols = ["order_id", "order_date", "promised_date", "delivered_date",
                 "on_time", "is_late", "order_cycle_days"]
    feature_cols = [c for c in feat.columns if c not in drop_cols]

    X = feat[feature_cols]
    y = feat["is_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    log_reg_probs = log_reg.predict_proba(X_test_scaled)[:, 1]

    rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    print("=== Logistic Regression ===")
    print(classification_report(y_test, log_reg.predict(X_test_scaled), zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, log_reg_probs):.3f}\n")

    print("=== Random Forest ===")
    print(classification_report(y_test, rf.predict(X_test), zero_division=0))
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_probs):.3f}\n")

    importances = pd.Series(rf.feature_importances_, index=feature_cols)
    print("Top 5 predictors of late delivery (Random Forest):")
    print(importances.sort_values(ascending=False).head(5))


if __name__ == "__main__":
    raw = _dc.load_data()
    cleaned = _dc.clean_orders(raw)
    features = build_features(cleaned)
    train_and_evaluate(features)
