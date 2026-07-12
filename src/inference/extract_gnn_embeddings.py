import os
import torch
import time

from src.models.graphsage import FraudGNN

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))
PYG_GRAPH_PATH = os.path.join(PROCESSED_DIR, "pyg_graph.pt")
MODEL_SAVE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../models_gnn.pth"))
EMBEDDINGS_SAVE_PATH = os.path.join(PROCESSED_DIR, "gnn_embeddings.pt")

def extract_embeddings():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Device: {device}")

    print("Loading graph data...")
    data = torch.load(PYG_GRAPH_PATH).to(device)

    print("Loading trained FraudGNN model...")
    # Initialize the architecture
    model = FraudGNN(num_node_features=data.num_node_features, hidden_dim=64, embedding_dim=64).to(device)
    
    # Load the weights you are currently training!
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("✅ Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {MODEL_SAVE_PATH}. Wait for training to finish!")
        return

    # Put model in evaluation mode (turns off dropout)
    model.eval()
    
    print(f"Extracting 64-dimensional embeddings for {data.num_nodes:,} nodes...")
    start_time = time.time()
    
    with torch.no_grad():
        # We ONLY call the encoder portion of the model
        embeddings = model.encoder(data.x, data.edge_index)
        
    # Move to CPU for saving and clustering later
    embeddings = embeddings.cpu()
    
    print("Saving embeddings...")
    torch.save(embeddings, EMBEDDINGS_SAVE_PATH)
    
    elapsed = time.time() - start_time
    print(f"✅ Extraction Complete in {elapsed:.2f} seconds!")
    print(f"  Saved to: {EMBEDDINGS_SAVE_PATH}")
    print(f"  Shape: {list(embeddings.shape)}")

if __name__ == "__main__":
    extract_embeddings()