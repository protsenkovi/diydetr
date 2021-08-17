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

dataset_root = Path("./coco_dataset/")
assert dataset_root.exists()

mode = 'instances'
PATHS = {
    "train": (dataset_root / "train2017", dataset_root / "annotations" / f'{mode}_train2017.json'),
    "val": (dataset_root / "val2017", dataset_root / "annotations" / f'{mode}_val2017.json')
}

img_folder, ann_file = PATHS["train"]
print("Training on", img_folder, ann_file )


epochs = 100000
device = torch.device('cuda')
batch_size = 8

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
    
  batch_tensors, batch_meta = transpose(batch)
  batch_tensor = uniform_shape_masked_tensor(batch_tensors)
  return batch_tensor, batch_meta

data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, collate_fn=collate_fn, pin_memory=True)
dataset_iterator = iter(data_loader)

diydetr = DIYDETR( 
  embed_dim=128, 
  num_classes=len(dataset.coco.getCatIds())+10, 
  num_object_slots=3, 
  dropout=0.0,
  number_of_resnet_layers=18
)
diydetr = diydetr.to(device)

optimizer = torch.optim.AdamW(diydetr.parameters())

f, (axes_images, axes_masks, axes_pred, axes_target_masks) = plt.subplots(4, batch_size, figsize=(batch_size*3, 4*3))
start_time = datetime.now()
for epoch in range(epochs):

  config.epoch = epoch

  try:
    inputs, targets = next(dataset_iterator)
  except StopIteration:
    dataset_iterator = iter(data_loader)
    inputs, targets = next(dataset_iterator)

  predicted = diydetr(inputs.to(device))
  loss = diydetr_loss(predicted, targets)
  config.tb.add_scalar("loss", loss, config.epoch)

  loss.backward()
  # for param_name, param_value in diydetr.named_parameters():
  #   config.tb.add_scalar(param_name + ".nan", param_value.isnan().sum(), epoch)
  #   config.tb.add_scalar(param_name + ".inf", param_value.isinf().sum(), epoch)
  #   config.tb.add_histogram(param_name, param_value, epoch)
  #   if param_value.grad is not None:
  #     config.tb.add_histogram(param_name + ".grad", param_value.grad, epoch)
  optimizer.step()
  print("\r{}/{}, elapsed: {}, loss: {:.4f}".format(epoch,epochs, datetime.now()-start_time, loss.detach()), end="")
  if epoch % 100 == 0:
    f = visualize(
      f, 
      axes_images, 
      axes_masks, 
      axes_pred,
      axes_target_masks,
      predicted=predicted, 
      inputs=inputs, 
      targets=targets)

    f.savefig('runs/result_current.png'.format(epoch))
    f.savefig('runs/result_{:06d}.png'.format(epoch))
    # f.canvas.draw()
    # config.tb.add_image("prediction_visualization", np.array(f.canvas.buffer_rgba()), config.epoch)
    [a.cla() for a in axes_images]
    [a.cla() for a in axes_masks]
    [a.cla() for a in axes_pred]
    [a.cla() for a in axes_target_masks]

config.tb.close()