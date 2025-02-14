from graph_transformer_pytorch import GraphTransformer
import torch
import torch.nn as nn
from GCN import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
  layers = []
  for i in range(len(dim_list) - 1):
    dim_in, dim_out = dim_list[i], dim_list[i + 1]
    layers.append(nn.Linear(dim_in, dim_out))
    final_layer = (i == len(dim_list) - 2)
    if not final_layer or final_nonlinearity:
      if batch_norm == 'batch':
        layers.append(nn.BatchNorm1d(dim_out))
      if activation == 'relu':
        layers.append(nn.ReLU())
      elif activation == 'leakyrelu':
        layers.append(nn.LeakyReLU())
    if dropout > 0:
      layers.append(nn.Dropout(p=dropout))
  return nn.Sequential(*layers)

class BoxRefinementNetwork(nn.Module):
  def __init__(self, emb_dim, hidden_dim, non_linearity = "LeakyReLU", dropout=0.1):
    super(BoxRefinementNetwork, self).__init__()
    box_net_dim = 4
    box_net_layers = [emb_dim, hidden_dim, box_net_dim]
    self.box_net = build_mlp(box_net_layers, batch_norm="batch")

  def forward(self, node_embeddings):
    if len(node_embeddings.shape) == 3: # process batch too
      batch, nodes, embedding = node_embeddings.shape
      node_embeddings = node_embeddings.view(-1, embedding)
      output = self.box_net(node_embeddings)
      output = output.view(batch, nodes, -1)

    elif len(node_embeddings.shape) == 2:
      output = self.box_net(node_embeddings)

    return output

class GraphTransformerBoxRefinementNetwork(nn.Module):
  def __init__(self, node_emb_dim, edge_emb_dim, transformer_depth, box_hidden_dim, box_nonlinearity="LeakyReLU", dropout=0.1, device=device):
    super(GraphTransformerBoxRefinementNetwork, self).__init__()
    self.graph_transformer = GraphTransformer(
        dim = node_emb_dim,
        edge_dim = edge_emb_dim,
        depth = transformer_depth,
        with_feedforwards = True,
        gated_residual = True,
        rel_pos_emb = True
    ).to(device)

    # self.graph_transformer = GCN(node_emb_dim, edge_emb_dim, 128).to(device)

    for p in self.graph_transformer.parameters(): 
      if p.dim() > 1:
        torch.nn.init.kaiming_uniform_(p, nonlinearity="relu")

    self.box_net = BoxRefinementNetwork(node_emb_dim, box_hidden_dim, box_nonlinearity, dropout).to(device)
    # box_net is already initialized with a distribution

  def forward(self, node_embeddings, edge_embeddings, masks):
    nodes, edges = self.graph_transformer(node_embeddings, edge_embeddings, mask = masks)

    boxes = self.box_net(nodes)

    return boxes
