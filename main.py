import torch
from models.diydetr import DIYDETR
from datasets.coco import CocoDetection
from datasets.transformations import ToTensor
from utils.functions import uniform_shape_masked_tensor
from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from utils.detr_loss import compute_loss
from pathlib import Path
import time
import matplotlib.pyplot as plt
from utils.visualize import visualize
from utils import config

dataset_root = Path("./coco_dataset/")
assert dataset_root.exists()

mode = 'instances'
PATHS = {
    "train": (dataset_root / "train2017", dataset_root / "annotations" / f'{mode}_train2017.json'),
    "val": (dataset_root / "val2017", dataset_root / "annotations" / f'{mode}_val2017.json')
}

img_folder, ann_file = PATHS["train"]
print("Training on", img_folder, ann_file )


epochs = 1000
device = torch.device('cuda')
batch_size = 3

dataset = CocoDetection(
    img_folder,
    ann_file,
    transforms=ToTensor(), #make_coco_transforms(image_set), 
    return_masks=True)
random_sampler = RandomSampler(dataset)
batch_sampler = BatchSampler(random_sampler, batch_size=batch_size, drop_last=True)

def collate_fn(batch):
  def transpose(iter_of_iterables):
    return list(zip(*iter_of_iterables))
    
  batch_tensors, batch_meta = transpose(batch)
  batch_tensor = uniform_shape_masked_tensor(batch_tensors)
  return batch_tensor, batch_meta

data_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0, collate_fn=collate_fn)
dataset_iterator = iter(data_loader)

diydetr = DIYDETR( embed_dim=16, num_classes=len(dataset.coco.getCatIds())+10, num_object_slots=100 )
diydetr = diydetr.to(device)

optimizer = torch.optim.AdamW(diydetr.parameters())

# torch.autograd.set_detect_anomaly(True)

f, (axes_images, axes_masks, axes_pred) = plt.subplots(3, batch_size, figsize=(batch_size*3, 3*3))

for epoch in range(epochs):
  # for param_name, param_value in diydetr.named_parameters():
  #   tb.add_histogram(param_name, param_value, epoch)
  config.epoch = epoch
  inputs, targets = next(dataset_iterator)
  predicted = diydetr(inputs.to(device))
  loss = compute_loss(predicted, targets)
  loss.backward()
  optimizer.step()
  print("\r{}/{}".format(epoch,epochs), end="")
  if epoch % 100 == 0:
    f = visualize(f, axes_images, axes_masks, axes_pred, predicted=predicted, inputs=inputs)

    f.savefig('result_current.png'.format(epoch))
    [a.cla() for a in axes_images]
    [a.cla() for a in axes_masks]
    [a.cla() for a in axes_pred]

config.tb.close()