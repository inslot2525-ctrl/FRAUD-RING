import torch
import numpy as np
from src.models.gnn_model import GraphSAGE

print("Loading graph...")
data = torch.load("data/processed/pyg_graph.pt", weights_only=False)

print("Loading trained model...")
model = GraphSAGE(in_channels=data.x.shape[1])
model.load_state_dict(torch.load("models_gnn.pth"))
model.eval()

with torch.no_grad():
    logits, embeddings = model(data.x, data.edge_index)

embeddings = embeddings.numpy()

np.save("data/processed/gnn_embeddings.npy", embeddings)

print("Embeddings shape:", embeddings.shape)
print("Saved embeddings.")