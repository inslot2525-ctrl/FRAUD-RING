"""
negative_sampling.py
--------------------
Generates negative (non-existent) edges for link-prediction training.

Strategy
--------
For each positive (fraud) edge we sample `neg_ratio` random edges that do
NOT exist in the graph.  This keeps the negative set proportional and avoids
trivially easy negatives.

Outputs
-------
data/processed/neg_edge_index.pt  – LongTensor [2, N_neg]
"""

import os
import pickle

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed"))


def load_edge_index(processed_dir: str) -> torch.Tensor:
    path = os.path.join(processed_dir, "edge_index.pt")
    edge_index = torch.load(path, weights_only=True)
    return edge_index


def sample_negatives(
    edge_index: torch.Tensor,
    num_nodes: int,
    neg_ratio: float = 1.0,
    seed: int = 42,
) -> torch.Tensor:
    """
    Parameters
    ----------
    edge_index : LongTensor [2, E]  – existing (positive) edges
    num_nodes  : total number of nodes in the graph
    neg_ratio  : negatives per positive edge
    seed       : RNG seed for reproducibility

    Returns
    -------
    neg_edge_index : LongTensor [2, N_neg]
    """
    torch.manual_seed(seed)

    # Build a set of existing edges for O(1) lookup
    existing = set(
        zip(edge_index[0].tolist(), edge_index[1].tolist())
    )

    num_pos = edge_index.size(1)
    num_neg = int(num_pos * neg_ratio)

    src_neg, dst_neg = [], []
    attempts = 0
    max_attempts = num_neg * 20  # safety cap

    while len(src_neg) < num_neg and attempts < max_attempts:
        attempts += 1
        s = torch.randint(0, num_nodes, (1,)).item()
        d = torch.randint(0, num_nodes, (1,)).item()
        if s != d and (s, d) not in existing:
            src_neg.append(s)
            dst_neg.append(d)
            existing.add((s, d))  # avoid duplicates within negatives

    sampled = len(src_neg)
    if sampled < num_neg:
        print(f"  Warning: only sampled {sampled}/{num_neg} negatives "
              f"after {attempts} attempts.")

    neg_edge_index = torch.tensor([src_neg, dst_neg], dtype=torch.long)
    print(f"  Sampled {neg_edge_index.size(1):,} negative edges "
          f"(ratio={neg_ratio})")
    return neg_edge_index


def main() -> None:
    print("Loading edge index...")
    edge_index = load_edge_index(PROCESSED_DIR)

    # Number of nodes = max index + 1
    num_nodes = int(edge_index.max().item()) + 1
    print(f"  Nodes: {num_nodes:,} | Positive edges: {edge_index.size(1):,}")

    neg_edge_index = sample_negatives(edge_index, num_nodes, neg_ratio=1.0)

    out_path = os.path.join(PROCESSED_DIR, "neg_edge_index.pt")
    torch.save(neg_edge_index, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
