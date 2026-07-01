import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

print("Loading embeddings...")
embeddings = np.load("data/processed/gnn_embeddings.npy")

print("Loading anomaly scores...")
anomaly_df = pd.read_csv("data/processed/anomaly_scores.csv")

# Keep only anomalous accounts
anomalies = anomaly_df[anomaly_df["is_anomaly"] == 1].copy()

print("Number of suspicious accounts:", len(anomalies))

# Get corresponding embeddings
anomaly_indices = anomalies.index.values
anomaly_embeddings = embeddings[anomaly_indices]

print("Running DBSCAN...")
dbscan = DBSCAN(
    eps=0.8,
    min_samples=3
)

clusters = dbscan.fit_predict(anomaly_embeddings)

anomalies["cluster"] = clusters

anomalies.to_csv("data/processed/fraud_clusters.csv", index=False)

print(anomalies.head(20))
print("\nCluster distribution:")
print(anomalies["cluster"].value_counts())

print("Saved fraud clusters.")