import pandas as pd
import networkx as nx
import os

# Resolve path relative to script directory to make it robust to where the script is run from
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/raw/paysim/paysim.csv"))


def load_data():
    print("Loading PaySim dataset...")
    df = pd.read_csv(DATA_PATH, nrows=100000)   # sample first
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


def main():
    df = load_data()
    inspect(df)

    G = build_transaction_graph(df)
    graph_stats(G)


if __name__ == "__main__":
    main()