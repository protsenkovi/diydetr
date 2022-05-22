import os
import torch
from models.diydetr import DIYDETR
from datasets.coco import CocoDetection, train_coco_transforms
from datasets.transformations import ToTensor
from utils.functions import uniform_shape_masked_tensor
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from diydetr_loss import diydetr_loss
from pathlib import Path
import time
from datetime import datetime
import matplotlib.pyplot as plt
from utils.visualize import visualize
from utils import config
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

dataset_root = Path("./coco_dataset/")
assert dataset_root.exists()

mode = 'instances'
PATHS = {
    "train": (dataset_root / "train2017", dataset_root / "annotations" / f'{mode}_train2017.json'),
    "val": (dataset_root / "val2017", dataset_root / "annotations" / f'{mode}_val2017.json')
}

img_folder, ann_file = PATHS["train"]
print("Training on", img_folder, ann_file )


epochs = int(1e8)
device = torch.device('cuda')
batch_size = 2


dataset = CocoDetection(
    img_folder,
    ann_file,
    transforms=train_coco_transforms(),
    return_masks=True)
random_sampler = RandomSampler(dataset)
batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=True)

def collate_fn(batch):
  def transpose(iter_of_iterables):
    return list(zip(*iter_of_iterables))
  # to pattern-match we need to 'transpose' a sequence of pairs to two sequences
  batch_tensors, batch_meta = transpose(batch) 
  batch_tensor = uniform_shape_masked_tensor(batch_tensors)
  return batch_tensor, batch_meta

data_loader = DataLoader(
  dataset=dataset, 
  batch_sampler=batch_sampler, 
  num_workers=2, 
  collate_fn=collate_fn, 
  pin_memory=False, 
  prefetch_factor=4
)
dataset_iterator = iter(data_loader)

if os.path.isfile('current_model.pt'):
  model = torch.load('current_model.pt')
  print("Continue")
else:
  model = DIYDETR( 
   embed_dim=256, 
   num_classes=config.num_classes, 
   num_object_slots=3, 
   dropout=0.1,
   number_of_resnet_layers=50,
   num_heads=8,
   device=device,
   num_transformer_layers=6
 )
 print("New model")
print()
print(model)
print()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable params count: {}".format(total_params))
print()

param_dicts = [
    {
      "params": [p for n, p in model.named_parameters() if "resnet" not in n and p.requires_grad],
      "lr": 1e-4
    },
    {
      "params": [p for n, p in model.named_parameters() if "resnet" in n and p.requires_grad],
      "lr": 1e-5,
    },

]
optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)

test_inputs, test_targets = next(dataset_iterator)

nrows = 4 if batch_size > 1 else 3
scale = 3
f, axs = plt.subplots(nrows, batch_size, figsize=(batch_size*scale, nrows*scale))
if batch_size == 1:
  axs = axs[:, None] 
  axes_images, axes_pred, axes_target_masks = axs
  axes_masks = None
else:
  axes_images, axes_masks, axes_pred, axes_target_masks = axs

start_time = datetime.now()
frame = 0

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#   with record_function("train_loop"): 
for epoch in range(epochs):
  config.epoch = epoch
  torch.cuda.empty_cache()
  try:
    inputs, targets = next(dataset_iterator)
  except StopIteration:
    dataset_iterator = iter(data_loader)
    inputs, targets = next(dataset_iterator)

  predicted = model(inputs.to(device))
  loss = diydetr_loss(predicted, targets)
  config.tb.add_scalar("loss/total", loss, config.epoch)

  loss.backward()
  # important for stable training
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-1)

  optimizer.step()
  print("\r{}/{}, elapsed: {}, loss: {:.4f}".format(epoch,epochs, datetime.now()-start_time, loss.detach()), end="")
  if epoch % 1000 == 0 and epoch > 0:
      torch.save(model, "current_model.pt")

  if epoch % 500 == 0:
    test_predicted = model(test_inputs.to(device))
    f = visualize(
      f, 
      axes_images, 
      axes_masks, 
      axes_pred,
      axes_target_masks,
      predicted=test_predicted, 
      inputs=test_inputs, 
      targets=test_targets)

    f.savefig('runs/result_current.png')
    f.savefig('runs/result_{:06d}.png'.format(frame))

    [a.cla() for a in axes_images]
    if axes_masks is not None:
      [a.cla() for a in axes_masks]
    [a.cla() for a in axes_pred]
    [a.cla() for a in axes_target_masks]

    f = visualize(
      f, 
      axes_images, 
      axes_masks, 
      axes_pred,
      axes_target_masks,
      predicted=predicted, 
      inputs=inputs, 
      targets=targets)

    f.savefig('runs/result_rand_current.png')
    f.savefig('runs/result_rand_{:06d}.png'.format(frame))

    [a.cla() for a in axes_images]
    if axes_masks is not None:
      [a.cla() for a in axes_masks]
    [a.cla() for a in axes_pred]
    [a.cla() for a in axes_target_masks]

    frame = frame + 1

# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
print()
config.tb.close()
