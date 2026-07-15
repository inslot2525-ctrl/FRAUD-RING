from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FraudGNN Live API", description="Continuous Graph Neural Network Engine")

# Configure CORS so your React frontend (Vite) can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# SIMULATED GNN DATABASE (Based on our exact PyTorch results)
# In production, these endpoints would query your PyTorch models or a Graph DB.
# =====================================================================

STATS = {
    "total_nodes": 9073900,
    "total_edges": 6362620,
    "known_fraudsters": 16382,
    "suspected_mules": 33349
}

# The top 5 threat networks discovered by MiniBatchKMeans
RINGS = [
    {"cluster_id": 102, "rank": 1, "total_accounts": 8214, "known_fraudsters": 8213, "suspected_mules": 1, "top_targets": ["C439737079"]},
    {"cluster_id": 125, "rank": 2, "total_accounts": 2685, "known_fraudsters": 2678, "suspected_mules": 7, "top_targets": ["M360644783", "M321063872"]},
    {"cluster_id": 469, "rank": 3, "total_accounts": 6444, "known_fraudsters": 7, "suspected_mules": 6437, "top_targets": ["C997608398"]},
    {"cluster_id": 284, "rank": 4, "total_accounts": 5484, "known_fraudsters": 5484, "suspected_mules": 0, "top_targets": []},
    {"cluster_id": 184, "rank": 5, "total_accounts": 26904, "known_fraudsters": 0, "suspected_mules": 26904, "top_targets": ["C1231006815"]}
]

# Specific deep-dive topology data for our suspected mules
INVESTIGATIONS = {
    "C439737079": {
        "risk_level": "CRITICAL RISK",
        "description": "Account sits at the exact mathematical center of 8,213 known fraudsters. Acting as a centralized sink (receiver) for micro-transactions.",
        "cluster_id": 102,
        "cluster_size": 8214,
        "fraud_ratio": 8213 / 8214,
        "graph_data": {
            "nodes": [
                {"id": "C439737079", "group": "mule"},
                {"id": "F_8831", "group": "fraud"},
                {"id": "F_9921", "group": "fraud"},
                {"id": "F_1023", "group": "fraud"},
                {"id": "F_4421", "group": "fraud"},
                {"id": "Victim_A", "group": "normal"},
                {"id": "Victim_B", "group": "normal"}
            ],
            "links": [
                {"source": "F_8831", "target": "C439737079"},
                {"source": "F_9921", "target": "C439737079"},
                {"source": "F_1023", "target": "C439737079"},
                {"source": "F_4421", "target": "C439737079"},
                {"source": "F_8831", "target": "F_9921"},
                {"source": "Victim_A", "target": "F_8831"},
                {"source": "Victim_B", "target": "F_1023"}
            ]
        }
    }
}

# =====================================================================
# REST ENDPOINTS
# =====================================================================

@app.get("/api/stats")
def get_stats():
    """Returns top-level scanning metrics."""
    return STATS


@app.get("/api/rings")
def get_rings(top_n: int = Query(10, description="Number of rings to return")):
    """Returns the most dangerous identified clusters."""
    return {"rings": RINGS[:top_n]}


@app.get("/api/investigate/{account_id}")
def investigate_account(account_id: str):
    """
    Returns specific risk details and network topology for a requested account.
    If the account isn't in our high-risk DB, return a clean profile.
    """
    account_id = account_id.strip()
    
    if account_id in INVESTIGATIONS:
        return INVESTIGATIONS[account_id]
        
    # Standard response for safe/unflagged accounts
    return {
        "risk_level": "LOW RISK",
        "description": "No topological proximity to known fraud clusters detected in the embedding space. Account behavior appears normal.",
        "cluster_id": "Unassigned",
        "cluster_size": 1,
        "fraud_ratio": 0.0,
        "graph_data": {
            "nodes": [{"id": account_id, "group": "normal"}],
            "links": []
        }
    }