from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, List, Dict, Tuple
import torch
from utils import GraphDataProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"

class GraphDataset(Dataset):
  def __init__(self,
    edge_lists, 
    ground_truth,
    id2names,
    image_ids,
    textual_encoder,
    node_emb_size: int = 300,
    edge_emb_size: int = 300):

    """
    Dataset for graph data.

    Args:
      edge_lists: contains information about edges encoded as edge lists 
      ground_truth: ground truth in the form of a list of (objectID, x, y, w, h) tuples
      id2names: dictionary that maps ids to names
      node_emb_size: dimension of node embeddings
      edge_emb_size: dimension of edge embeddings
    """


    self.edge_lists = []
    self.ground_truths = []
    self.image_ids = []
    self.id2names = []

    # handle images for which there are no relationships (example: image 51)
    for i in range(len(edge_lists)):
      if len(ground_truth[i]) == 0:
        continue
      else:
        self.edge_lists.append(edge_lists[i])
        self.ground_truths.append(ground_truth[i])
        self.id2names.append(id2names[i])
        self.image_ids.append(image_ids[i])

    
    self.processor = GraphDataProcessor(
      embedding_dim=node_emb_size,
      edge_dim=edge_emb_size,
      node_encoder=textual_encoder,
      edge_encoder=textual_encoder,
    )

  def __len__(self) -> int:
    return len(self.edge_lists)

  def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    edge_list = self.edge_lists[idx]
    ground_truth = self.ground_truths[idx]
    id2name = self.id2names[idx]

    graph_data = self.processor.process_edge_list(edge_list, ground_truth, id2name)
    graph_data['image_ids'] = self.image_ids[idx]

    return graph_data


def collate_graphs(batch):
  batch_size = len(batch)

  sizes = [b['nodes'].shape[0] for b in batch]
  max_nodes = max(sizes)

  # nodes tensor
  nodes = torch.zeros((batch_size, max_nodes, batch[0]['nodes'].shape[1]))
  for i, b in enumerate(batch):
    nodes[i, :b['nodes'].shape[0]] = b['nodes']
  
  edges = torch.zeros((batch_size, max_nodes, max_nodes, batch[0]['edges'].shape[2]))
  for i, b in enumerate(batch):
    edges[i, : b['edges'].shape[0], : b['edges'].shape[1], :] = b['edges']

  masks = torch.zeros((batch_size, max_nodes), dtype=torch.bool)
  for i, b in enumerate(batch):
    masks[i, : b['mask'].shape[0]] = b['mask']

  ground_truths = [b['ground_truth'] for b in batch]

  return {
      'nodes': nodes,
      'edges': edges,
      'masks': masks,
      'ground_truths': ground_truths,
      'image_ids': [batch[i]['image_ids'] for i in range(batch_size)]
  }

def create_graph_dataloader(
    edge_lists, 
    ground_truth,
    id2names,
    image_ids,
    text_encoder_fn,
    batch_size = 32,
    shuffle = True,
    embedding_size = 300 # emb size for both nodes and edges
):
    """
    Create a DataLoader for graph data.

    Args:
        edge_lists: List of edge lists for each graph
        ground_truths: list of ground truths for each object
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle the data
        embedding_size: embedding size for nodes and edges

    Returns:
        DataLoaders (train and test) for the graph data
    """

    dataset = GraphDataset(
      edge_lists = edge_lists,
      ground_truth = ground_truth,
      id2names = id2names,
      image_ids = image_ids,
      textual_encoder = text_encoder_fn,
      node_emb_size = embedding_size,
      edge_emb_size = embedding_size
    )

    percentage_test = 0.2
    test_size = int(percentage_test * len(dataset))
    train_size = len(dataset) - test_size 
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle = shuffle, collate_fn = collate_graphs)
    test_dataloader = DataLoader(test_set, batch_size = batch_size, shuffle = False, collate_fn = collate_graphs)

    return train_dataloader, test_dataloader
