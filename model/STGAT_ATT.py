import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from models.MultiRAU import MultiHeadRAU

class STGATModuleWithRAU(nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, last_layer_output, global_pooling):
        super(STGATModuleWithRAU, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.last_layer_output = last_layer_output
        self.global_pooling = global_pooling

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
            heads=1,
            edge_dim=3
        )

        # Here we replace GRU with our simple RAU.
        self.rau = MultiHeadRAU(input_size=hidden_channels, hidden_size=256)

        # Fully connected layer for classification
        self.fc = nn.Linear(256, last_layer_output)

    def forward(self, X, edge_index, edge_attr):
        """
        X: shape (seq_length, n_nodes, in_channels)
        edge_index, edge_attr: graph structure
        """
        seq_length, n_nodes, _ = X.shape

        # (1) Extract node embeddings for each time step via GAT
        time_embeddings = []
        for t in range(seq_length):
            X_t = X[t]  # shape: (n_nodes, in_channels)
            # Spatial steps
            X_t = self.gat1(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)
            X_t = self.gat2(X_t, edge_index, edge_attr)
            X_t = F.elu(X_t)
            time_embeddings.append(X_t)  # shape: (n_nodes, hidden_channels)

        # (2) Stack => (seq_length, n_nodes, hidden_channels)
        node_embeddings = torch.stack(time_embeddings, dim=0)
        # Permute => (n_nodes, seq_length, hidden_channels)
        node_embeddings = node_embeddings.permute(1, 0, 2)

        # (3) Manually unroll RAU over the temporal dimension
        # Initialize hidden states for each node
        # (batch_size = n_nodes, hidden_size = 256)
        h = self.rau.init_hidden(batch_size=n_nodes).to(node_embeddings.device)

        for t in range(seq_length):
            x_t = node_embeddings[:, t, :]  # shape: (n_nodes, hidden_channels)
            h = self.rau(x_t, h)            # shape: (n_nodes, 256)

        # After unrolling, 'h' is the final hidden state for each node: (n_nodes, 256)

        # (4) Optionally do graph-level pooling
        if self.global_pooling:
            # shape => (1, 256) after mean pooling across n_nodes
            h_mean = h.mean(dim=0, keepdim=True)
            out = self.fc(h_mean)  # (1, last_layer_output)
        else:
            # Per-node output => shape: (n_nodes, last_layer_output)
            out = self.fc(h)

        return out
