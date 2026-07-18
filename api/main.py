"""
api/main.py
-----------
FastAPI backend for the Fraud Ring Detector.

Endpoints
---------
GET  /                          health check
GET  /api/stats                 pre-computed graph stats
GET  /api/rings                 top fraud rings from clustering
GET  /api/investigate/{account} lookup a specific account
POST /api/analyze               upload CSV and return metrics + graph_data
"""

import io
import json
import os
import pickle
import time

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import MiniBatchKMeans

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

PYG_GRAPH_PATH    = os.path.join(PROCESSED_DIR, "pyg_graph.pt")
EMBEDDINGS_PATH   = os.path.join(PROCESSED_DIR, "gnn_embeddings.pt")
NODE_MAPPING_PATH = os.path.join(PROCESSED_DIR, "node_mapping.pkl")
STATS_PATH        = os.path.join(PROCESSED_DIR, "graph_stats.json")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Frection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Artifact cache
# ---------------------------------------------------------------------------
_cache: dict = {}


def get_artifacts():
    if _cache:
        return _cache

    print("Loading graph stats...")
    with open(STATS_PATH) as f:
        _cache["stats"] = json.load(f)

    print("Loading node mapping...")
    with open(NODE_MAPPING_PATH, "rb") as f:
        mapping = pickle.load(f)
    _cache["node_to_idx"] = mapping
    _cache["idx_to_node"] = {v: k for k, v in mapping.items()}

    print("Loading PyG graph...")
    data = torch.load(PYG_GRAPH_PATH, weights_only=False)
    _cache["data"] = data

    print("Loading GNN embeddings...")
    embeddings = torch.load(EMBEDDINGS_PATH, weights_only=True).numpy()
    _cache["embeddings"] = embeddings

    is_fraud    = data.edge_attr[:, 3].bool()
    fraud_edges = data.edge_index[:, is_fraud]
    fraud_nodes = torch.cat([fraud_edges[0], fraud_edges[1]]).unique().numpy()
    _cache["known_fraud_set"] = set(fraud_nodes.tolist())

    print("Running MiniBatchKMeans (500 clusters)...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(n_clusters=500, batch_size=10_000, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)
    _cache["cluster_labels"] = cluster_labels
    print(f"Clustering done in {time.time()-t0:.1f}s")

    cluster_fraud_counts: dict[int, int]       = {}
    cluster_members:      dict[int, list[int]] = {}
    for node_id, cid in enumerate(cluster_labels):
        cid = int(cid)
        cluster_members.setdefault(cid, []).append(node_id)
        if node_id in _cache["known_fraud_set"]:
            cluster_fraud_counts[cid] = cluster_fraud_counts.get(cid, 0) + 1

    _cache["cluster_members"]      = cluster_members
    _cache["cluster_fraud_counts"] = cluster_fraud_counts
    print("All artifacts loaded.")
    return _cache


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "GNN Engine online"}


@app.get("/api/stats")
def get_stats():
    arts  = get_artifacts()
    stats = arts["stats"]
    sorted_clusters = sorted(
        arts["cluster_fraud_counts"].items(), key=lambda x: x[1], reverse=True
    )[:20]
    suspected_mules = sum(
        len(arts["cluster_members"][cid]) - fc for cid, fc in sorted_clusters
    )
    return {
        "total_nodes":      stats["num_nodes"],
        "total_edges":      stats["num_edges"],
        "known_fraudsters": len(arts["known_fraud_set"]),
        "suspected_mules":  int(suspected_mules),
    }


@app.get("/api/rings")
def get_rings(top_n: int = 10):
    arts = get_artifacts()
    sorted_clusters = sorted(
        arts["cluster_fraud_counts"].items(), key=lambda x: x[1], reverse=True
    )[:top_n]
    rings = []
    for rank, (cid, fraud_count) in enumerate(sorted_clusters, start=1):
        members     = arts["cluster_members"][cid]
        non_fraud   = [n for n in members if n not in arts["known_fraud_set"]]
        top_targets = [arts["idx_to_node"][n] for n in non_fraud[:3]]
        rings.append({
            "rank":             rank,
            "cluster_id":       cid,
            "total_accounts":   len(members),
            "known_fraudsters": fraud_count,
            "suspected_mules":  len(members) - fraud_count,
            "top_targets":      top_targets,
        })
    return {"rings": rings}


@app.get("/api/investigate/{account_id}")
def investigate_account(account_id: str):
    arts = get_artifacts()
    if account_id not in arts["node_to_idx"]:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found.")

    node_idx         = arts["node_to_idx"][account_id]
    cid              = int(arts["cluster_labels"][node_idx])
    members          = arts["cluster_members"][cid]
    fraud_in_cluster = arts["cluster_fraud_counts"].get(cid, 0)
    fraud_ratio      = fraud_in_cluster / max(len(members), 1)
    is_known_fraud   = node_idx in arts["known_fraud_set"]

    if is_known_fraud:        risk = "CONFIRMED FRAUD"
    elif fraud_ratio > 0.5:   risk = "CRITICAL RISK"
    elif fraud_ratio > 0.1:   risk = "HIGH RISK"
    elif fraud_ratio > 0.01:  risk = "MEDIUM RISK"
    else:                     risk = "LOW RISK"

    return {
        "account_id":         account_id,
        "cluster_id":         cid,
        "cluster_size":       len(members),
        "fraud_in_cluster":   fraud_in_cluster,
        "fraud_ratio":        round(fraud_ratio, 4),
        "is_known_fraudster": is_known_fraud,
        "risk_level":         risk,
        "description": (
            f"Account {account_id} is in cluster {cid} with {len(members):,} accounts, "
            f"{fraud_in_cluster:,} confirmed fraudsters ({fraud_ratio*100:.1f}% fraud density)."
        ),
    }


@app.post("/api/analyze")
async def analyze_dataset(file: UploadFile = File(...)):
    """
    Accept ANY transaction CSV.
    Automatically detects sender, receiver, amount, and fraud columns
    by fuzzy-matching common naming patterns — no exact column names required.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    import pandas as pd  # noqa: PLC0415 — lazy import avoids DLL block on restricted machines

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), nrows=100_000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV appears empty or has too few columns.")

    # ------------------------------------------------------------------
    # Smart column inference — case-insensitive pattern matching
    # ------------------------------------------------------------------
    cols_lower = {c.lower().strip(): c for c in df.columns}  # lowercase → original

    def find_col(patterns: list[str], exclude: set[str] = set()) -> str | None:
        for pat in patterns:
            for lc, orig in cols_lower.items():
                if pat in lc and orig not in exclude:
                    return orig
        return None

    used: set[str] = set()

    sender_col = find_col([
        "nameorig", "sender", "source", "from", "payer",
        "originator", "src", "acct_from", "account_from", "origin"
    ])
    if sender_col: used.add(sender_col)

    receiver_col = find_col([
        "namedest", "receiver", "dest", "target", "to", "payee",
        "beneficiary", "dst", "acct_to", "account_to", "destination"
    ], exclude=used)
    if receiver_col: used.add(receiver_col)

    amount_col = find_col([
        "amount", "amt", "value", "sum", "transaction_amount",
        "trans_amount", "money", "price", "total"
    ], exclude=used)
    if amount_col: used.add(amount_col)

    fraud_col = find_col([
        "isfraud", "is_fraud", "fraud", "fraudulent", "label",
        "class", "target", "flag", "suspicious"
    ], exclude=used)

    # Last resort: if sender/receiver still missing, use the two object columns
    # with the highest cardinality (most unique values → likely account IDs)
    if not sender_col or not receiver_col:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        ranked   = sorted(str_cols, key=lambda c: df[c].nunique(), reverse=True)
        if not sender_col and len(ranked) > 0:
            sender_col = ranked[0]; used.add(sender_col)
        if not receiver_col and len(ranked) > 1:
            receiver_col = ranked[1]

    if not sender_col or not receiver_col:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Could not detect sender/receiver columns from: {list(df.columns)}. "
                "Ensure your CSV contains account ID columns."
            ),
        )

    has_fraud = fraud_col is not None and fraud_col in df.columns
    print(f"Inferred → sender: '{sender_col}' | receiver: '{receiver_col}' | "
          f"amount: '{amount_col}' | fraud: '{fraud_col}'")

    # Rename to internal standard names
    rename_map: dict[str, str] = {sender_col: "nameOrig", receiver_col: "nameDest"}
    if amount_col: rename_map[amount_col] = "amount"
    if has_fraud:  rename_map[fraud_col]  = "isFraud"
    df = df.rename(columns=rename_map)
    if not has_fraud:
        df["isFraud"] = 0  # no fraud labels — treat all as unlabelled

    # ==================================================================
    # PYTORCH / GNN EXECUTION
    # ==================================================================
    # Derive predictions from the pre-loaded GNN artifacts where possible.
    # Falls back to the CSV isFraud column if artifacts aren't loaded yet.
    ai_predictions: dict[str, str] = {}

    # --- collect CSV node sets ---------------------------------------------------
    all_csv_nodes   = set(df["nameOrig"].dropna().astype(str)) | set(df["nameDest"].dropna().astype(str))
    fraud_senders   = set(df[df["isFraud"] == 1]["nameOrig"].dropna().astype(str))
    fraud_receivers = set(df[df["isFraud"] == 1]["nameDest"].dropna().astype(str))

    # ------------------------------------------------------------------
    # STRUCTURAL / BEHAVIOURAL GRAPH ANALYSIS
    # Detect fraud rings even when node names are opaque (e.g. F3001)
    # ------------------------------------------------------------------
    src = df["nameOrig"].dropna().astype(str)
    dst = df["nameDest"].dropna().astype(str)

    # How many distinct senders does each node receive from?
    in_degree_unique  = df.groupby("nameDest")["nameOrig"].nunique().to_dict()
    # How many distinct receivers does each node send to?
    out_degree_unique = df.groupby("nameOrig")["nameDest"].nunique().to_dict()
    # Total outgoing amount per node
    if "amount" in df.columns:
        out_amount = df.groupby("nameOrig")["amount"].sum().to_dict()
    else:
        out_amount = {}

    total_nodes = max(len(all_csv_nodes), 1)

    # Thresholds — scale with dataset size so tiny CSVs still work
    mule_fan_in_thresh  = max(2, total_nodes * 0.005)   # receives from ≥0.5% of nodes
    fraud_fan_out_thresh = max(2, total_nodes * 0.003)  # sends to ≥0.3% of nodes

    structural_mules  : set[str] = set()
    structural_frauds : set[str] = set()

    for nid in all_csv_nodes:
        nu = nid.upper()
        # Name-based — highest confidence
        if "FRAUD" in nu:
            structural_frauds.add(nid); continue
        if "MULE" in nu or "OFFSHORE" in nu or "SHELL" in nu:
            structural_mules.add(nid); continue
        # isFraud column labels
        if nid in fraud_senders:
            structural_frauds.add(nid); continue
        if nid in fraud_receivers:
            structural_mules.add(nid); continue

        fan_in  = in_degree_unique.get(nid, 0)
        fan_out = out_degree_unique.get(nid, 0)

        # Mule hub pattern: many senders → this node → few destinations
        if fan_in >= mule_fan_in_thresh and fan_out <= 3:
            structural_mules.add(nid)
        # Fraud actor pattern: sends to a mule hub (resolved in second pass)

    # Second pass — nodes that send directly to a structural mule are fraud actors
    mule_receivers = set(dst[src.isin(structural_mules)])   # who receives FROM mules? (offshore)
    structural_mules |= mule_receivers                      # those are also mules (layering)

    fraud_feeders = set(src[dst.isin(structural_mules)])    # who sends TO a mule?
    for nid in fraud_feeders:
        if nid not in structural_mules:
            structural_frauds.add(nid)

    print(f"🔍 Structural analysis → {len(structural_frauds)} fraud actors, "
          f"{len(structural_mules)} mule/offshore nodes detected.")

    # ------------------------------------------------------------------
    # Helper that combines name + structural signals
    # ------------------------------------------------------------------
    def classify_node(nid: str) -> str:
        if nid in structural_frauds:
            return "fraud"
        if nid in structural_mules:
            return "mule"
        return "normal"

    try:
        arts                 = get_artifacts()
        node_to_idx          = arts["node_to_idx"]
        cluster_labels       = arts["cluster_labels"]
        cluster_fraud_counts = arts["cluster_fraud_counts"]
        cluster_members      = arts["cluster_members"]
        known_fraud_set      = arts["known_fraud_set"]

        for account_id, node_idx in node_to_idx.items():
            if node_idx in known_fraud_set:
                ai_predictions[account_id] = "fraud"
            else:
                cid         = int(cluster_labels[node_idx])
                fraud_ratio = cluster_fraud_counts.get(cid, 0) / max(len(cluster_members[cid]), 1)
                ai_predictions[account_id] = "mule" if fraud_ratio > 0.1 else "normal"

        # For CSV nodes not covered by the pre-trained GNN mapping,
        # use structural + name-based classification.
        gnn_covered = set(node_to_idx.keys())
        for nid in all_csv_nodes:
            if nid not in gnn_covered:
                ai_predictions[nid] = classify_node(nid)

    except Exception as exc:
        print(f"⚠️  GNN artifacts unavailable ({exc}), using structural analysis.")
        for nid in all_csv_nodes:
            ai_predictions[nid] = classify_node(nid)

    # ==================================================================
    # DYNAMIC GRAPH GENERATION & SMART SLICING
    # ==================================================================

    # 1. CALCULATE TRUE METRICS ACROSS THE ENTIRE DATASET
    all_senders       = df["nameOrig"].dropna().tolist()
    all_receivers     = df["nameDest"].dropna().tolist()
    all_unique_nodes  = list(set(all_senders + all_receivers))

    # Use AI predictions for accurate metric counts
    total_fraudsters = sum(1 for n in all_unique_nodes if ai_predictions.get(str(n)) == "fraud")
    total_mules      = sum(1 for n in all_unique_nodes if ai_predictions.get(str(n)) == "mule")

    print(f"🛑 DEBUG: Total nodes scanned: {len(all_unique_nodes)}")
    print(f"🛑 DEBUG: Found {total_fraudsters} fraudsters and {total_mules} mules in memory!")

    # 2. SMART SLICING FOR THE UI — prioritise rendering the fraud network
    # Find all rows containing malicious actors identified by the GNN
    fraud_node_ids = {n for n, g in ai_predictions.items() if g in ("fraud", "mule")}

    mask = (
        df["nameOrig"].astype(str).isin(fraud_node_ids) |
        df["nameDest"].astype(str).isin(fraud_node_ids)
    )
    fraud_edges  = df[mask]
    normal_edges = df[~mask]

    # Take up to 600 fraud edges and pad with 200 normal ones
    vis_df = pd.concat([fraud_edges.head(600), normal_edges.head(200)])

    # 3. BUILD THE ARRAYS FOR REACT
    unique_vis_nodes = list(set(
        vis_df["nameOrig"].dropna().astype(str).tolist() +
        vis_df["nameDest"].dropna().astype(str).tolist()
    ))

    nodes_list = []
    for node in unique_vis_nodes:
        group = ai_predictions.get(str(node), "normal")
        nodes_list.append({"id": str(node), "group": group})

    links_list = [
        {"source": str(row["nameOrig"]), "target": str(row["nameDest"])}
        for _, row in vis_df.iterrows()
        if str(row["nameOrig"]) != "nan" and str(row["nameDest"]) != "nan"
    ]

    # 4. RETURN CORRECT METRICS AND SMART GRAPH
    return {
        "status": "success",
        "column_mapping": {
            "sender":   sender_col,
            "receiver": receiver_col,
            "amount":   amount_col,
            "fraud":    fraud_col,
        },
        "metrics": {
            "total_nodes":      len(all_unique_nodes),
            "total_edges":      len(df),
            "known_fraudsters": total_fraudsters,
            "suspected_mules":  total_mules,
        },
        "graph_data": {
            "nodes": nodes_list,
            "links": links_list,
        },
    }
