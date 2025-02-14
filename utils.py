import torch
from typing import List, Tuple, Dict, Callable
import numpy as np
import json
from tqdm import tqdm

import fasttext

model = fasttext.load_model('cc.en.300.bin')
def get_word_embedding(word):
    vector = model.get_word_vector(word)
    tensor = torch.tensor(vector, dtype=torch.float32)
    return tensor

device = "cuda" if torch.cuda.is_available() else "cpu"

def process_graph(sg):
  relationships = sg['relationships']
  objects = sg['objects']

  edge_list = []
  gt = []

  ids = set()

  for r in relationships:
    object_id = r["object_id"]
    subject_id = r["subject_id"]
    edge_list.append((object_id, subject_id, r["predicate"]))
    
    ids.add(object_id)
    ids.add(subject_id)
  
  for o in objects: # only process objects that are in relationships
    if o['object_id'] in ids:
      gt.append((o['object_id'], o['x'], o['y'], o['w'], o['h']))

  id2name = {o['object_id']: o['names'][0] for o in sg['objects']}
  
  return edge_list, gt, id2name, sg['image_id']

class GraphDataProcessor:
  def __init__(self,
    node_encoder,
    edge_encoder,
    embedding_dim: int = 256,
    edge_dim: int = 512):
    """
    Initialize Graph Data Processor with encoders for nodes and edges.

    Args:
      node_encoder: Function that converts node strings to embedding vectors
      edge_encoder: Function that converts relationship strings to edge vectors
      embedding_dim: Dimension of node embeddings
      edge_dim: Dimension of edge features
    """
    self.embedding_dim = embedding_dim
    self.edge_dim = edge_dim
    self.node_encoder = node_encoder
    self.edge_encoder = edge_encoder

  def process_edge_list(self, edges, ground_truth, id2name):
    """
    Convert edge list to format suitable for graph transformers.
    Also returns ground truth ordered in the same way as the nodes in the nodes matrix

    Args:
      edges: List of tuples (objectID, subjectID, relationship)
      ground_truth: list of (objectID, x, y, w, h)
    Returns:
      Dict containing 'nodes', 'edges', 'mask' and 'ground_truth' tensors
    """

    # get unique nodes
    unique_nodes = []
    seen = set()
    for obj, subj, _ in edges:
      for node in (obj, subj):
        if node not in seen:
          seen.add(node)
          unique_nodes.append(node)

    num_nodes = len(unique_nodes)

    # order them by id
    unique_nodes.sort()
    
    # encode objects by name
    node_names = [id2name[n] for n in unique_nodes]

    # Encode node names
    nodes = torch.zeros(num_nodes, self.embedding_dim)
    for i, n in enumerate(node_names):
      nodes[i] = self.node_encoder(n)

    # Encode edges
    edges_tensor = torch.zeros(num_nodes, num_nodes, self.edge_dim)

    # Ensure correct mapping between positions in edge matrix and nodes
    node_positions = {node: idx for idx, node in enumerate(unique_nodes)}

    # Create edge features
    for obj, subj, rel in edges:
      obj_idx = node_positions[obj]
      subj_idx = node_positions[subj]
      edge_feature = self.edge_encoder(rel)

      edges_tensor[obj_idx, subj_idx] = edge_feature

    mask = torch.ones(num_nodes).bool()

    # Order ground truth by id
    ground_truth.sort(key = lambda x: x[0])
    ground_truth = [(id2name[t[0]], t[1], t[2], t[3], t[4]) for t in ground_truth]

    return {
      'nodes': nodes,
      'edges': edges_tensor,
      'mask': mask,
      'ground_truth': ground_truth
    }

def read_scene_graphs(filename):
  with open(filename, "r") as f:
    data = json.load(f)

  edge_lists = []
  ground_truths = []
  id2names = []
  image_ids = []

  for d in data:
    edges, gts, id2name, image_id = process_graph(d)
    edge_lists.append(edges)
    ground_truths.append(gts)
    id2names.append(id2name)
    image_ids.append(image_id)

  return edge_lists, ground_truths, id2names, image_ids, {d['image_id']: d for d in data}

def read_resolutions(filename):
  with open(filename, "r") as f:
    data = json.load(f)

  return {d['image_id']: (d['width'], d['height']) for d in data}

def normalize(coords, resolution):
  x, y, w, h = coords
  maxw, maxh = resolution
  return x/maxw, y/maxh, w/maxw, h/maxh

def denormalize(coords, resolution):
  x, y, w, h = coords
  maxw, maxh = resolution
  return x * maxw, y * maxh, w * maxw, h * maxh

def normalize_coords(ground_truths, image_ids, resolutions):
  new_gt = []
  for i in range(len(image_ids)):
    gt = ground_truths[i]
    resolution = resolutions[image_ids[i]]

    for j, g in enumerate(gt):
      coords = (g[1], g[2], g[3], g[4])
      coords = normalize(coords, resolution)

      gt[j] = (g[0], *coords)
      
    new_gt.append(gt)
  return new_gt

def denormalize_coords(ground_truths, image_ids, resolutions):
  new_gt = []
  for i in range(len(image_ids)):
    gt = ground_truths[i]
    resolution = resolutions[image_ids[i]]

    for j, g in enumerate(gt):
      coords = (g[1], g[2], g[3], g[4])
      coords = denormalize(coords, resolution)

      gt[j] = (g[0], *coords)
      
    new_gt.append(gt)
  return new_gt

def IoU_loss(hyp, ref):
  """
  Computes IoU loss between predicted and target bounding boxes.
  Args:
      hyp: Tensor of shape (N, 4) with (x, y, w, h) format containing layouts as predicted by the model
      ref: Tensor of shape (N, 4) with (x, y, w, h) format containing ground truth layouts
  Returns:
      IoU loss (scalar tensor)
  """
  # Convert (x, y, w, h) to (x1, y1, x2, y2)
  hyp_x1 = hyp[:, 0]
  hyp_y1 = hyp[:, 1]
  hyp_x2 = hyp_x1 + hyp[:, 2]
  hyp_y2 = hyp_y1 + hyp[:, 3]
  ref_x1 = ref[:, 0]
  ref_y1 = ref[:, 1]
  ref_x2 = ref_x1 + ref[:, 2]
  ref_y2 = ref_y1 + ref[:, 3]
  
  # Compute intersection box (top-left and bottom-right)
  inter_x1 = torch.max(hyp_x1, ref_x1)
  inter_y1 = torch.max(hyp_y1, ref_y1)
  inter_x2 = torch.min(hyp_x2, ref_x2)
  inter_y2 = torch.min(hyp_y2, ref_y2)
  
  # Compute intersection area
  inter_w = (inter_x2 - inter_x1).clamp(min=0)
  inter_h = (inter_y2 - inter_y1).clamp(min=0)
  intersection = inter_w * inter_h
  
  # Compute areas of both boxes
  hyp_area = (hyp_x2 - hyp_x1) * (hyp_y2 - hyp_y1)
  ref_area = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
  
  # Compute union area
  eps = 1e-10
  union = hyp_area + ref_area - intersection + eps
  
  # Compute IoU
  iou = intersection / union
  
  # IoU loss
  loss = 1 - iou.mean()  # Mean over batch
  return loss

def combined_loss(hyp, ref):
  gamma = 0.5
  return gamma * IoU_loss(hyp, ref) + torch.nn.functional.mse_loss(hyp, ref)
