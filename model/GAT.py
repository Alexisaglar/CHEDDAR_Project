import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import GRU, Linear

class GATModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, last_layer_output, heads):
        super(GATModule, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.last_layer_output = last_layer_output

        # GAT layers for spatial dependencies
        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=3
        )
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=3
        )

        # Fully connected layer for per-node classification
        self.fc = Linear(hidden_channels * heads, last_layer_output)  # Multiply by 2 for bidirectional GRU

    def forward(self, data):
        # n_nodes, _ = data.shape
        X, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Apply GAT layers
        X = self.gat1(X, edge_index, edge_attr)
        X = F.elu(X)
        X = self.gat2(X, edge_index, edge_attr)
        X = F.elu(X)

        out = self.fc(X)  # Shape: (33, n_classes)
        # print(out.shape)

        return out
