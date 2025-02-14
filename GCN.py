import torch
import torch.nn as nn
import torch.nn.functional as F

class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        self.layer_norm = nn.LayerNorm(node_dim)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_uniform_(p)
    
    def forward(self, nodes, edges, mask):
        batch_size, num_nodes, node_dim = nodes.shape
        
        # Create attention mask for message passing
        # Shape: (batch, nodes, nodes)
        # This mask will be True for valid sender->receiver pairs
        message_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        
        # Prepare node features for message computation
        nodes_expanded = nodes.unsqueeze(2)
        
        # Concatenate node and edge features
        message_input = torch.cat([
            nodes_expanded.expand(-1, -1, num_nodes, -1),
            edges
        ], dim=-1)
        
        # Compute messages
        messages = self.message_fn(message_input)
        
        # Apply message mask
        # Shape: (batch, nodes, nodes, hidden_dim)
        messages = messages * message_mask.unsqueeze(-1)
        
        # Aggregate messages (sum over neighboring nodes)
        # Only valid messages will be summed due to masking
        aggregated = messages.sum(dim=2)
        
        # Update node features
        update_input = torch.cat([nodes, aggregated], dim=-1)
        updates = self.update_fn(update_input)
        
        # Apply node mask to updates
        updates = updates * mask.unsqueeze(-1)
        
        # Residual connection and layer norm (only for valid nodes)
        nodes = self.layer_norm(nodes + updates) * mask.unsqueeze(-1)
        
        return nodes

class GCN(nn.Module):
    def __init__(
        self, 
        node_dim, 
        edge_dim, 
        hidden_dim=256, 
        num_layers=3, 
        dropout=0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MessagePassingLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        self.output_projection = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, nodes, edges, mask):
        """
        Args:
            nodes: Node features tensor (batch, nodes, node_dim)
            edges: Edge features tensor (batch, nodes, nodes, edge_dim)
            mask: Boolean mask tensor (batch, nodes) where True indicates valid nodes
        """
        # Initial mask application
        nodes = nodes * mask.unsqueeze(-1)
        
        # Create edge mask from node mask
        edge_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        edges = edges * edge_mask.unsqueeze(-1)
        
        # Message passing layers
        for layer in self.layers:
            nodes = self.dropout(layer(nodes, edges, mask))
        
        # Final projection (only for valid nodes)
        nodes = self.output_projection(nodes) * mask.unsqueeze(-1)
        
        return nodes