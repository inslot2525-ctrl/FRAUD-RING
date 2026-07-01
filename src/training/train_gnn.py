import torch
from src.models.gnn_model import GraphSAGE

print("Loading graph...")
data = torch.load("data/processed/pyg_graph.pt", weights_only=False)

model = GraphSAGE(in_channels=data.x.shape[1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# Dummy labels for now (unsupervised placeholder)
labels = torch.zeros(data.x.shape[0], 1)

epochs = 20

for epoch in range(epochs):
    optimizer.zero_grad()

    logits, embeddings = model(data.x, data.edge_index)

    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}")

torch.save(model.state_dict(), "models_gnn.pth")
print("GNN model saved.")