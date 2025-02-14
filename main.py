import torch
from model import GraphTransformerBoxRefinementNetwork
from dataset import *
from utils import *
from training import *
device = "cuda" if torch.cuda.is_available() else "cpu"

# creating dataset and dataloaders
edge_lists, ground_truths, id2names, image_ids, id2scenegraph = read_scene_graphs("./scene_graphs.json")
resolutions = read_resolutions("./image_data.json")
ground_truths = normalize_coords(ground_truths, image_ids, resolutions)

train_dataloader, test_dataloader = create_graph_dataloader(
    edge_lists = edge_lists,
    ground_truth = ground_truths,
    id2names = id2names,
    image_ids = image_ids,
    batch_size = 1,
    text_encoder_fn = get_word_embedding,
    shuffle = False
)

print("created dataloader")

# instantiating model
model = GraphTransformerBoxRefinementNetwork(
    node_emb_dim = 300,
    edge_emb_dim = 300,
    transformer_depth = 2,
    box_hidden_dim = 128,
    device=device
)

loss_fn = combined_loss
training_loop(model, train_dataloader, loss_fn = loss_fn, epochs = 5, lr = 1e-4)

loss_eval = IoU_loss
print(f"evaluation: {eval_loop(model, test_dataloader, loss_fn = loss_eval)}")
