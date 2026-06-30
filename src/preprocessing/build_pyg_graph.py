import torch
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
import pickle


print("Loading node features...")
df = pd.read_csv("data/processed/node_features.csv")

# Remove non-feature columns
feature_cols = [
    col for col in df.columns
    if col not in ["account", "fraud_count", "fraud_ratio"]
]

# Node feature matrix
x = torch.tensor(df[feature_cols].values, dtype=torch.float)

print("Loading graph...")
with open("data/processed/graph.pkl", "rb") as f:
    G = pickle.load(f)

print("Creating node mapping...")
node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}

print("Building edge index...")
edges = []

for src, dst in G.edges():
    if src in node_to_idx and dst in node_to_idx:
        edges.append([node_to_idx[src], node_to_idx[dst]])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

data = Data(x=x, edge_index=edge_index)

torch.save(data, "data/processed/pyg_graph.pt")

print(data)
print("Saved PyG graph.")