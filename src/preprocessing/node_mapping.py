"""
node_mapping.py
---------------
Streams the full PaySim CSV in chunks and builds:

  - A string-to-integer node mapping  (account ID → contiguous int index)
  - edge_index.pt   [2, E]  LongTensor  (src/dst integer node IDs)
  - edge_attr.pt    [E, 5]  FloatTensor (amount, step, tx_type, isFraud, isFlaggedFraud)
  - node_mapping.pkl        dict {account_str: int}
  - graph_stats.json        quick sanity-check stats

Processing in 500 k-row chunks keeps RAM well under 2 GB even for the full
6.36 M-row dataset.
"""

import json
import os
import pickle

import pandas as pd
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../data/raw/paysim/paysim.csv")
)
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))

EDGE_INDEX_PATH   = os.path.join(PROCESSED_DIR, "edge_index.pt")
EDGE_ATTR_PATH    = os.path.join(PROCESSED_DIR, "edge_attr.pt")
NODE_MAPPING_PATH = os.path.join(PROCESSED_DIR, "node_mapping.pkl")
STATS_PATH        = os.path.join(PROCESSED_DIR, "graph_stats.json")

# PaySim transaction-type strings → integers
TX_TYPE_MAP = {
    "PAYMENT":   0,
    "TRANSFER":  1,
    "CASH_OUT":  2,
    "DEBIT":     3,
    "CASH_IN":   4,
}

CHUNK_SIZE         = 500_000
TOTAL_ROWS_APPROX  = 6_362_620   # used only for the tqdm progress bar


def build_node_mapping_and_edges() -> None:
    print(f"Reading dataset: {DATA_PATH}")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    node_mapping: dict[str, int] = {}
    node_counter = 0

    src_nodes:        list[int]   = []
    dst_nodes:        list[int]   = []
    amounts:          list[float] = []
    steps:            list[int]   = []
    tx_types:         list[int]   = []
    is_fraud:         list[int]   = []
    is_flagged_fraud: list[int]   = []

    with tqdm(total=TOTAL_ROWS_APPROX, desc="Processing chunks", unit="rows") as pbar:
        for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
            for row in chunk.itertuples(index=False):
                sender   = row.nameOrig
                receiver = row.nameDest

                # Assign contiguous integer IDs on first sight
                if sender not in node_mapping:
                    node_mapping[sender] = node_counter
                    node_counter += 1
                if receiver not in node_mapping:
                    node_mapping[receiver] = node_counter
                    node_counter += 1

                src_nodes.append(node_mapping[sender])
                dst_nodes.append(node_mapping[receiver])

                amounts.append(float(row.amount))
                steps.append(int(row.step))
                tx_types.append(TX_TYPE_MAP.get(row.type, -1))
                is_fraud.append(int(row.isFraud))
                is_flagged_fraud.append(int(row.isFlaggedFraud))

            pbar.update(len(chunk))

    num_edges = len(src_nodes)
    print(f"\nUnique nodes : {node_counter:,}")
    print(f"Total edges  : {num_edges:,}")
    print(f"Fraud edges  : {sum(is_fraud):,} ({sum(is_fraud)/num_edges*100:.3f}%)")

    # ------------------------------------------------------------------
    # Build tensors
    # ------------------------------------------------------------------
    print("\nConverting to PyTorch tensors...")

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    edge_attr = torch.tensor(
        list(zip(amounts, steps, tx_types, is_fraud, is_flagged_fraud)),
        dtype=torch.float32,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    print("Saving to disk...")
    torch.save(edge_index, EDGE_INDEX_PATH)
    torch.save(edge_attr,  EDGE_ATTR_PATH)

    with open(NODE_MAPPING_PATH, "wb") as f:
        pickle.dump(node_mapping, f)

    stats = {
        "num_nodes":       node_counter,
        "num_edges":       num_edges,
        "num_fraud_edges": sum(is_fraud),
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=4)

    print("\n✅ Stage 1 complete. Files saved:")
    print(f"  edge_index.pt   shape: {list(edge_index.shape)}")
    print(f"  edge_attr.pt    shape: {list(edge_attr.shape)}")
    print(f"  node_mapping.pkl")
    print(f"  graph_stats.json")
    print(f"\nAll outputs in: {PROCESSED_DIR}")


if __name__ == "__main__":
    build_node_mapping_and_edges()
