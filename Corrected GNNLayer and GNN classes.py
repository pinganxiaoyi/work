# Corrected GNNLayer and GNN classes
import torch
import torch.nn as nn
import torch.nn.functional as F
# Define a single GNN layer
class GNNLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(GNNLayer, self).__init__()
        
        # Node feature transformation
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)  # Output dimension matches input dimension
        )
        
        # Edge feature transformation
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_dim)  # Output dimension matches input dimension
        )
        
    def forward(self, x_nodes, x_edges):
        # Update node features
        new_x_nodes = self.node_mlp(x_nodes)
        
        # Update edge features
        new_x_edges = self.edge_mlp(x_edges)
        
        return new_x_nodes, new_x_edges

# Define the overall GNN model
class GNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, L):
        super(GNN, self).__init__()
        
        self.L = L
        self.gnn_layer = GNNLayer(node_dim, edge_dim, hidden_dim)
        
    def forward(self, x_nodes, x_edges):
        for _ in range(self.L):
            x_nodes, x_edges = self.gnn_layer(x_nodes, x_edges)
        return x_nodes, x_edges

# Initialize the model and test it with some random data
node_dim = 5  # Node feature dimension
edge_dim = 6  # Edge feature dimension
hidden_dim = 10  # Hidden layer dimension
L = 3  # Number of GNN layers (iterations)

model = GNN(node_dim, edge_dim, hidden_dim, L)

# Generate some example node and edge features
x_nodes = torch.rand((10, node_dim))  # 10 nodes with features of dimension node_dim
x_edges = torch.rand((20, edge_dim))  # 20 edges with features of dimension edge_dim

# Forward pass through the GNN model
out_nodes, out_edges = model(x_nodes, x_edges)

out_nodes, out_edges
print("Output nodes:", out_nodes)
print("Output edges:", out_edges)