import os
import torch
import time

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
PYG_GRAPH_PATH = os.path.join(PROCESSED_DIR, "pyg_graph.pt")

def balance_training_edges():
    print("Loading split PyG graph...")
    start_time = time.time()
    
    data = torch.load(PYG_GRAPH_PATH, weights_only=False)
    
    # Extract training mask and fraud labels
    train_mask = data.train_edge_mask
    is_fraud = data.edge_attr[:, 3].bool()
    
    # 1. Find indices of all Train Fraud and Train Normal edges
    train_fraud_indices = torch.where(train_mask & is_fraud)[0]
    train_normal_indices = torch.where(train_mask & ~is_fraud)[0]
    
    num_fraud = train_fraud_indices.size(0)
    
    # 2. Sample Normal edges at a 1:10 ratio
    ratio = 10
    num_normal_to_sample = num_fraud * ratio
    
    print(f"Found {num_fraud:,} Fraud edges in training.")
    print(f"Sampling {num_normal_to_sample:,} Normal edges (1:{ratio} ratio)...")
    
    # Randomly shuffle and select
    perm = torch.randperm(train_normal_indices.size(0))
    sampled_normal_indices = train_normal_indices[perm[:num_normal_to_sample]]
    
    # 3. Combine them to create our Training Supervision Set
    supervision_indices = torch.cat([train_fraud_indices, sampled_normal_indices])
    
    # Shuffle the final combined list so fraud/normal are mixed during training
    supervision_indices = supervision_indices[torch.randperm(supervision_indices.size(0))]
    
    # 4. Save indices directly into the PyG Data object
    data.train_supervision_indices = supervision_indices
    
    # For Val and Test, we grade the model on EVERYTHING in that time period
    data.val_supervision_indices = torch.where(data.val_edge_mask)[0]
    data.test_supervision_indices = torch.where(data.test_edge_mask)[0]
    
    print("\nSupervision Sets Created:")
    print(f"  - Train: {data.train_supervision_indices.size(0):,} edges used for loss")
    print(f"  - Val:   {data.val_supervision_indices.size(0):,} edges used for evaluation")
    print(f"  - Test:  {data.test_supervision_indices.size(0):,} edges used for evaluation")
    
    # Save the updated graph
    torch.save(data, PYG_GRAPH_PATH)
    
    elapsed = time.time() - start_time
    print(f"\n✅ Negative Sampling Complete in {elapsed:.2f} seconds!")

if __name__ == "__main__":
    balance_training_edges()