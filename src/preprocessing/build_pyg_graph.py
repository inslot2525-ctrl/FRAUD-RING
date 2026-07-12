import os
import torch
from torch_geometric.data import Data
import time

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))

EDGE_INDEX_PATH = os.path.join(PROCESSED_DIR, "edge_index.pt")
EDGE_ATTR_PATH = os.path.join(PROCESSED_DIR, "edge_attr.pt")
NODE_FEAT_PATH = os.path.join(PROCESSED_DIR, "x_node_features.pt")
PYG_GRAPH_PATH = os.path.join(PROCESSED_DIR, "pyg_graph.pt")

def build_pyg_graph():
    print("Loading processed tensors...")
    start_time = time.time()
    
    # 1. Load the loose tensors
    edge_index = torch.load(EDGE_INDEX_PATH)
    edge_attr = torch.load(EDGE_ATTR_PATH)
    x = torch.load(NODE_FEAT_PATH)
    
    print("Constructing PyTorch Geometric Data object...")
    
    # 2. Package into a PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # 3. Validate properties
    print(f"\nGraph Properties:")
    print(f"  - Nodes: {data.num_nodes:,}")
    print(f"  - Edges: {data.num_edges:,}")
    print(f"  - Node Features: {data.num_node_features}")
    print(f"  - Edge Features: {data.num_edge_features}")
    print(f"  - Has Isolated Nodes: {data.has_isolated_nodes()}")
    print(f"  - Is Directed: {data.is_directed()}")
    
    # 4. Save the final payload
    print(f"\nSaving unified PyG graph...")
    torch.save(data, PYG_GRAPH_PATH)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Final Preprocessing Stage Complete in {elapsed:.2f} seconds!")
    print(f"  Saved to: {PYG_GRAPH_PATH}")

if __name__ == "__main__":
    build_pyg_graph()