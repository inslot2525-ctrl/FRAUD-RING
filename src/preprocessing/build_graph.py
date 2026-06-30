import pandas as pd
import networkx as nx
import os
import pickle

# Resolve path relative to script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../data/raw/paysim/paysim.csv")
)

GRAPH_SAVE_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../data/processed/graph.pkl")
)


def load_data():
    print("Loading PaySim dataset...")
    df = pd.read_csv(DATA_PATH, nrows=100000)
    return df


def inspect(df):
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFraud Distribution:")
    print(df["isFraud"].value_counts())


def build_transaction_graph(df):
    print("\nBuilding graph...")

    G = nx.DiGraph()

    for _, row in df.iterrows():
        sender = row["nameOrig"]
        receiver = row["nameDest"]

        G.add_node(sender)
        G.add_node(receiver)

        G.add_edge(
            sender,
            receiver,
            amount=row["amount"],
            step=row["step"],
            fraud=row["isFraud"]
        )

    return G


def graph_stats(G):
    print("\nGraph Statistics")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())


def save_graph(G):
    os.makedirs(os.path.dirname(GRAPH_SAVE_PATH), exist_ok=True)

    with open(GRAPH_SAVE_PATH, "wb") as f:
        pickle.dump(G, f)

    print(f"\nGraph saved to: {GRAPH_SAVE_PATH}")


def main():
    df = load_data()
    inspect(df)

    G = build_transaction_graph(df)
    graph_stats(G)
    save_graph(G)


if __name__ == "__main__":
    main()