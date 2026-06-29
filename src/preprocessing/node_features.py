import os
import pandas as pd
import networkx as nx

DATA_PATH = "data/raw/paysim/paysim.csv"


def load_data():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, nrows=100000)  # sample for now
    return df


def build_graph(df):
    print("Building graph...")
    G = nx.DiGraph()

    for _, row in df.iterrows():
        sender = row["nameOrig"]
        receiver = row["nameDest"]

        G.add_edge(
            sender,
            receiver,
            amount=row["amount"],
            fraud=row["isFraud"]
        )

    return G


def compute_node_features(G):
    features = []

    for node in G.nodes():
        in_degree = G.in_degree(node)
        out_degree = G.out_degree(node)

        incoming_edges = list(G.in_edges(node, data=True))
        outgoing_edges = list(G.out_edges(node, data=True))

        total_received = sum(data["amount"] for _, _, data in incoming_edges)
        total_sent = sum(data["amount"] for _, _, data in outgoing_edges)

        avg_received = total_received / in_degree if in_degree > 0 else 0
        avg_sent = total_sent / out_degree if out_degree > 0 else 0

        fraud_count = 0

        for _, _, data in incoming_edges:
            fraud_count += data["fraud"]

        for _, _, data in outgoing_edges:
            fraud_count += data["fraud"]

        total_edges = in_degree + out_degree
        fraud_ratio = fraud_count / total_edges if total_edges > 0 else 0

        features.append([
            node,
            in_degree,
            out_degree,
            total_sent,
            total_received,
            avg_sent,
            avg_received,
            fraud_count,
            fraud_ratio
        ])

    feature_df = pd.DataFrame(features, columns=[
        "account",
        "in_degree",
        "out_degree",
        "total_sent",
        "total_received",
        "avg_sent",
        "avg_received",
        "fraud_count",
        "fraud_ratio"
    ])

    return feature_df


def main():
    df = load_data()
    G = build_graph(df)
    feature_df = compute_node_features(G)

    os.makedirs("data/processed", exist_ok=True)
    feature_df.to_csv("data/processed/node_features.csv", index=False)

    print(feature_df.head())
    print("\nShape:", feature_df.shape)
    print("Saved node features.")


if __name__ == "__main__":
    main()