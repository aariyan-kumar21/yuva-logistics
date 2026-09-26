"""
04_clustering_delivery_zones.py
--------------------------------
Stage 3 (continued): Predictive/Descriptive Modelling — Clustering.

Groups delivery zones by demand volume, average shipping cost, and average
cycle time using K-Means. The resulting clusters help decide which zones
are candidates for a dedicated micro-fulfillment hub, which should be
served by third-party carriers, and which should be prioritized in route
optimization (Stage 4).

Run: python 04_clustering_delivery_zones.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.append(os.path.dirname(__file__))
from importlib import import_module
_dc = import_module("01_data_cleaning")


def build_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    zone_features = df.groupby("dest_zone").agg(
        total_orders=("order_id", "count"),
        avg_order_value=("order_value", "mean"),
        avg_shipping_cost=("shipping_cost", "mean"),
        avg_cycle_time=("order_cycle_days", "mean"),
        on_time_rate=("on_time", "mean"),
    ).reset_index()
    return zone_features


def cluster_zones(zone_features: pd.DataFrame, k_range=range(2, 7)):
    feature_cols = ["total_orders", "avg_order_value", "avg_shipping_cost",
                     "avg_cycle_time", "on_time_rate"]
    X = zone_features[feature_cols]
    X_scaled = StandardScaler().fit_transform(X)

    # simple silhouette-score sweep to choose k
    best_k, best_score = None, -1
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_k, best_score = k, score

    final_model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    zone_features = zone_features.copy()
    zone_features["cluster"] = final_model.fit_predict(X_scaled)

    return zone_features, best_k, best_score


if __name__ == "__main__":
    raw = _dc.load_data()
    cleaned = _dc.clean_orders(raw)

    zones = build_zone_features(cleaned)
    clustered, best_k, score = cluster_zones(zones)

    print(f"Chosen number of clusters (by silhouette score): {best_k} (score={score:.3f})\n")
    print(clustered.groupby("cluster")[
        ["total_orders", "avg_order_value", "avg_shipping_cost",
         "avg_cycle_time", "on_time_rate"]
    ].mean().round(2))

    print(
        "\nInterpretation: clusters with high total_orders + low on_time_rate are "
        "prime candidates for a dedicated micro-fulfillment hub or a renegotiated "
        "carrier SLA; clusters with low order volume are better served by "
        "consolidated, lower-frequency shipments."
    )
