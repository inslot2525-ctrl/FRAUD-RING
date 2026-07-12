import os
import time
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from src.models.graphsage import FraudGNN

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
PYG_GRAPH_PATH = os.path.join(PROCESSED_DIR, "pyg_graph.pt")
MODEL_SAVE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../models_gnn.pth"))

def train():
    # 1. Device Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}")

    # 2. Load Data
    print("Loading graph data...")
    data = torch.load(PYG_GRAPH_PATH, weights_only=False).to(device)

    # Extract labels (is_fraud is column 3 of edge_attr)
    labels = data.edge_attr[:, 3].float()

    # 3. Initialize Model, Optimizer, and Loss
    model = FraudGNN(num_node_features=data.num_node_features, hidden_dim=64, embedding_dim=64).to(device)
    
    # We use Adam optimizer with weight decay (L2 regularization) to prevent overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    
    # BCEWithLogitsLoss combines Sigmoid + Binary Cross Entropy for numerical stability
    # pos_weight upweights fraud class proportional to the 1:10 sampling ratio
    train_labels_all = labels[data.train_supervision_indices]
    n_neg = (train_labels_all == 0).sum().float()
    n_pos = (train_labels_all == 1).sum().float()
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Class balance — fraud: {n_pos:.0f} | normal: {n_neg:.0f} | pos_weight: {pos_weight.item():.2f}")

    epochs = 15
    best_val_pr_auc = 0.0
    patience = 5          # stop if no improvement for this many epochs
    patience_counter = 0

    print("\nStarting Training...")
    print("=" * 60)
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val ROC-AUC':<15} | {'Val PR-AUC':<15}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # --- TRAINING PHASE ---
        model.train()
        optimizer.zero_grad()
        
        # We only pass gradients for the balanced training supervision set
        train_indices = data.train_supervision_indices
        train_edges = data.edge_index[:, train_indices]
        train_labels = labels[train_indices]
        
        # Forward pass (Computes embeddings for ALL nodes, but only predicts for train_edges)
        logits = model(data.x, data.edge_index, train_edges)
        
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()
        
        # --- VALIDATION PHASE ---
        model.eval()
        with torch.no_grad():
            val_indices = data.val_supervision_indices
            val_edges = data.edge_index[:, val_indices]
            val_labels = labels[val_indices].cpu().numpy()
            
            val_logits = model(data.x, data.edge_index, val_edges)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            
            # Calculate metrics
            val_roc_auc = roc_auc_score(val_labels, val_probs)
            val_pr_auc = average_precision_score(val_labels, val_probs)
            
        elapsed = time.time() - start_time
        
        # Save best model
        saved_marker = ""
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            saved_marker = "⭐ Saved"
            patience_counter = 0
        else:
            patience_counter += 1
            
        print(f"{epoch:<6} | {loss.item():<12.4f} | {val_roc_auc:<15.4f} | {val_pr_auc:<15.4f} {saved_marker}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (no improvement for {patience} epochs).")
            break

    print("=" * 60)
    print(f"Training Complete! Best Validation PR-AUC: {best_val_pr_auc:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()