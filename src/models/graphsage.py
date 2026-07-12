"""
graphsage.py
------------
GraphSAGE encoder that produces node embeddings.

The encoder is kept separate from any task-specific head so the same
backbone can be reused by the edge decoder, the anomaly detector, etc.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGEEncoder(torch.nn.Module):
    """
    Two-layer GraphSAGE encoder.

    Parameters
    ----------
    input_dim     : number of input node features
    hidden_dim    : width of the first SAGE layer
    embedding_dim : output embedding size (per node)
    dropout       : dropout probability applied between layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : FloatTensor [N, input_dim]
        edge_index : LongTensor  [2, E]

        Returns
        -------
        embeddings : FloatTensor [N, embedding_dim]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        # No activation here — the edge decoder or downstream task decides
        return x
