import os
import torch
import time

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
PYG_GRAPH_PATH = os.path.join(PROCESSED_DIR, "pyg_graph.pt")

def split_edges_chronologically():
    print(f"Loading PyG graph from {PYG_GRAPH_PATH}...")
    start_time = time.time()
    
    # Load the unified graph
    data = torch.load(PYG_GRAPH_PATH, weights_only=False)
    
    # Extract the 'step' feature (Time hour) - it is at index 1 in edge_attr
    # edge_attr layout: [amount, step, tx_type, is_fraud, is_flagged_fraud]
    steps = data.edge_attr[:, 1]
    is_fraud = data.edge_attr[:, 3].bool()
    
    max_step = steps.max().item()
    print(f"Dataset spans {max_step} hours (steps).")
    
    # Define thresholds for 70% / 15% / 15%
    train_threshold = int(max_step * 0.70)
    val_threshold = int(max_step * 0.85)
    
    print(f"Splitting chronologically:")
    print(f"  - Train: steps 1 to {train_threshold}")
    print(f"  - Val:   steps {train_threshold + 1} to {val_threshold}")
    print(f"  - Test:  steps {val_threshold + 1} to {int(max_step)}")
    
    # Create Boolean Masks
    data.train_edge_mask = (steps <= train_threshold)
    data.val_edge_mask = (steps > train_threshold) & (steps <= val_threshold)
    data.test_edge_mask = (steps > val_threshold)
    
    # Validate the split
    train_count = data.train_edge_mask.sum().item()
    val_count = data.val_edge_mask.sum().item()
    test_count = data.test_edge_mask.sum().item()
    total = data.num_edges
    
    print("\nSplit Statistics:")
    print(f"  - Train Edges: {train_count:,} ({train_count/total*100:.1f}%) | Fraud: {is_fraud[data.train_edge_mask].sum().item():,}")
    print(f"  - Val Edges:   {val_count:,} ({val_count/total*100:.1f}%) | Fraud: {is_fraud[data.val_edge_mask].sum().item():,}")
    print(f"  - Test Edges:  {test_count:,} ({test_count/total*100:.1f}%) | Fraud: {is_fraud[data.test_edge_mask].sum().item():,}")
    
    # Save the updated graph with masks included
    print("\nSaving updated graph...")
    torch.save(data, PYG_GRAPH_PATH)
    
    elapsed = time.time() - start_time
    print(f"✅ Edge Splitting Complete in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    split_edges_chronologically()