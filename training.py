import torch
from torch.optim import Adam
from utils import *

def print_compare(layouts, ground_truths):
  lengths = [len(g) for g in ground_truths]
  for i in range(layouts.shape[0]): # bacth size:
    for j in range(lengths[i]):
      print(f"layout: {layouts[i][j].tolist()}, ground_truth: {ground_truths[i][j]}")
    print("-------------------------------------")

def training_loop(model, dataloader, loss_fn, epochs = 1, lr=4e-5): 
  print("training starting")
  model.train()
  # transformer_parameters = [p for p in model.graph_transformer.parameters()]
  # box_parameters = [p for p in model.box_net.parameters()]
  
  # optim_tr = Adam(transformer_parameters, lr=transformer_lr)
  # optim_box = Adam(box_parameters, lr=box_lr)

  optim = Adam([p for p in model.parameters()], lr = lr)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  losses = []
  for e in range(epochs):
    for i, batch in enumerate(dataloader):
      nodes = batch['nodes'].to(device)
      edges = batch['edges'].to(device)
      masks = batch['masks'].to(device)
      ground_truths = batch['ground_truths']
      ground_truth_tensor = []

      layouts = model(nodes, edges, masks)
      lengths = [len(g) for g in ground_truths]
      loss = torch.tensor([0.0], dtype=torch.float, device=device)
      for j, l in enumerate(lengths):
        gt_coords = [[gt[1], gt[2], gt[3], gt[4]] for gt in ground_truths[j]]        
        gt_coords = torch.tensor(gt_coords, device=device)

        loss += loss_fn(layouts[j, :l].squeeze(0), gt_coords)

      loss = loss / nodes.shape[0] # average over batches

      # optim_tr.zero_grad()
      # optim_box.zero_grad()
      optim.zero_grad()
      
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)


      if i % 100 == 0 or i == 0:
        print(f"[*] loss: {loss.item()}, epoch: {e}")
        losses.append(loss.item()) 

        for p in model.graph_transformer.parameters():
          if p.grad != None:
            if torch.isnan(p.grad).any():
              print("NaN detected in gradients!")

        # debugging grads
        # grads_tr = [torch.mean(p.grad) for p in transformer_parameters if p.grad != None]
        # grads_box = [torch.mean(p.grad) for p in box_parameters if p.grad != None]

        # print("-----------------------------------")
        # print(f"mean grads, transformer: {sum(grads_tr) / len(grads_tr)}, box: {sum(grads_box) / len(grads_box)}")
        # print("-----------------------------------")
        # print()
        # print()

      # if i % 1000 == 0:
      #   print("---------------------------------------")
      #   print_compare(layouts, ground_truths)
      #   print("---------------------------------------")
      #   print()

      optim.step()

  return losses

def eval_loop(model, dataloader, loss_fn):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  losses = []
  with torch.no_grad():
    for i, batch in enumerate(dataloader):
      nodes = batch['nodes'].to(device)
      edges = batch['edges'].to(device)
      masks = batch['masks'].to(device)
      ground_truths = batch['ground_truths']
      ground_truth_tensor = []

      layouts = model(nodes, edges, masks)
      lengths = [len(g) for g in ground_truths]
      loss = torch.tensor([0.0], dtype=torch.float, device=device)
      for j, l in enumerate(lengths):
        gt_coords = [[gt[1], gt[2], gt[3], gt[4]] for gt in ground_truths[j]]        
        gt_coords = torch.tensor(gt_coords, device=device)

        loss += loss_fn(layouts[j, :l].squeeze(0), gt_coords)
      loss = loss / nodes.shape[0] 
      
      losses.append(loss.item())
  return sum(losses) / len(losses)

    













     