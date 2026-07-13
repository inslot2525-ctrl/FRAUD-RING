"""
api/main.py
-----------
FastAPI backend for the Fraud Ring Detector.

Endpoints
---------
GET  /                          health check
GET  /api/stats                 pre-computed graph stats (fast)
GET  /api/rings                 top fraud rings from clustering
GET  /api/investigate/{account} lookup a specific account
POST /api/analyze               (future) upload a new CSV and rerun pipeline
"""

import json
import os
import pickle
import time

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import MiniBatchKMeans

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

PYG_GRAPH_PATH     = os.path.join(PROCESSED_DIR, "pyg_graph.pt")
EMBEDDINGS_PATH    = os.path.join(PROCESSED_DIR, "gnn_embeddings.pt")
NODE_MAPPING_PATH  = os.path.join(PROCESSED_DIR, "node_mapping.pkl")
STATS_PATH         = os.path.join(PROCESSED_DIR, "graph_stats.json")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="FraudGNN API", description="Graph Neural Network Fraud Detection Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: load heavy artifacts once into memory
# ---------------------------------------------------------------------------
_cache: dict = {}

def get_artifacts():
    """Load and cache all inference artifacts on first call."""
    if _cache:
        return _cache

    print("Loading graph stats...")
    with open(STATS_PATH) as f:
        _cache["stats"] = json.load(f)

    print("Loading node mapping...")
    with open(NODE_MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
    _cache["node_to_idx"]   = mapping
    _cache["idx_to_node"]   = {v: k for k, v in mapping.items()}

    print("Loading PyG graph...")
    data = torch.load(PYG_GRAPH_PATH, weights_only=False)
    _cache["data"] = data

    print("Loading GNN embeddings...")
    embeddings = torch.load(EMBEDDINGS_PATH, weights_only=True).numpy()
    _cache["embeddings"] = embeddings

    # Build fraud node set from edge labels
    is_fraud = data.edge_attr[:, 3].bool()
    fraud_edges = data.edge_index[:, is_fraud]
    fraud_nodes = torch.cat([fraud_edges[0], fraud_edges[1]]).unique().numpy()
    _cache["known_fraud_set"] = set(fraud_nodes.tolist())

    print("Running MiniBatchKMeans clustering (500 clusters)...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=500, batch_size=10_000, random_state=42, n_init="auto"
    )
    cluster_labels = kmeans.fit_predict(embeddings)
    _cache["cluster_labels"] = cluster_labels
    print(f"Clustering done in {time.time()-t0:.1f}s")

    # Pre-compute per-cluster stats
    cluster_fraud_counts: dict[int, int]        = {}
    cluster_members:      dict[int, list[int]]  = {}
    for node_id, cid in enumerate(cluster_labels):
        cid = int(cid)
        cluster_members.setdefault(cid, []).append(node_id)
        if node_id in _cache["known_fraud_set"]:
            cluster_fraud_counts[cid] = cluster_fraud_counts.get(cid, 0) + 1

    _cache["cluster_members"]      = cluster_members
    _cache["cluster_fraud_counts"] = cluster_fraud_counts

    print("✅ All artifacts loaded.")
    return _cache


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "GNN Engine online"}


@app.get("/api/stats")
def get_stats():
    """Return high-level graph statistics."""
    arts = get_artifacts()
    stats = arts["stats"]
    known_fraud_nodes = len(arts["known_fraud_set"])

    # Suspected mules: nodes in the top-20 fraud clusters that are NOT known fraudsters
    sorted_clusters = sorted(
        arts["cluster_fraud_counts"].items(), key=lambda x: x[1], reverse=True
    )[:20]
    suspected_mules = sum(
        len(arts["cluster_members"][cid]) - fraud_count
        for cid, fraud_count in sorted_clusters
    )

    return {
        "total_nodes":      stats["num_nodes"],
        "total_edges":      stats["num_edges"],
        "known_fraudsters": known_fraud_nodes,
        "suspected_mules":  int(suspected_mules),
    }


@app.get("/api/rings")
def get_rings(top_n: int = 10):
    """Return the top N fraud rings ranked by known-fraudster count."""
    arts   = get_artifacts()
    idx_to = arts["idx_to_node"]
    sorted_clusters = sorted(
        arts["cluster_fraud_counts"].items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    rings = []
    for rank, (cid, fraud_count) in enumerate(sorted_clusters, start=1):
        members       = arts["cluster_members"][cid]
        total         = len(members)
        mules         = total - fraud_count
        non_fraud_ids = [n for n in members if n not in arts["known_fraud_set"]]
        top_targets   = [idx_to[n] for n in non_fraud_ids[:3]]

        rings.append({
            "rank":             rank,
            "cluster_id":       cid,
            "total_accounts":   total,
            "known_fraudsters": fraud_count,
            "suspected_mules":  mules,
            "top_targets":      top_targets,
        })

    return {"rings": rings}


@app.get("/api/investigate/{account_id}")
def investigate_account(account_id: str):
    """Return risk profile for a specific account ID."""
    arts = get_artifacts()

    if account_id not in arts["node_to_idx"]:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found in graph.")

    node_idx = arts["node_to_idx"][account_id]
    cid      = int(arts["cluster_labels"][node_idx])
    members  = arts["cluster_members"][cid]
    fraud_in_cluster = arts["cluster_fraud_counts"].get(cid, 0)
    fraud_ratio      = fraud_in_cluster / max(len(members), 1)
    is_known_fraud   = node_idx in arts["known_fraud_set"]

    if is_known_fraud:
        risk = "CONFIRMED FRAUD"
    elif fraud_ratio > 0.5:
        risk = "CRITICAL RISK"
    elif fraud_ratio > 0.1:
        risk = "HIGH RISK"
    elif fraud_ratio > 0.01:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    return {
        "account_id":         account_id,
        "node_index":         node_idx,
        "cluster_id":         cid,
        "cluster_size":       len(members),
        "fraud_in_cluster":   fraud_in_cluster,
        "fraud_ratio":        round(fraud_ratio, 4),
        "is_known_fraudster": is_known_fraud,
        "risk_level":         risk,
        "description": (
            f"This account belongs to cluster {cid} containing {len(members):,} accounts, "
            f"of which {fraud_in_cluster:,} are confirmed fraudsters "
            f"({fraud_ratio*100:.1f}% fraud concentration)."
        ),
    }
