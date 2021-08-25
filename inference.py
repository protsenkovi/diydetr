import os
import cv2
import numpy as np
import time
import torch
from datasets.coco import inference_coco_transforms
from utils.functions import uniform_shape_masked_tensor

count = 1
start_time = time.time()

output_to_directory = True

vc = cv2.VideoCapture("mts0.mp4")

fps = int(vc.get(cv2.CAP_PROP_FPS))
width = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frame_number = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc('A', 'V', 'C', '1')

vw = cv2.VideoWriter("mts0_diydetr.mp4", fourcc, fps, (width, height))

model = torch.load("current_model.pt")

device = torch.device("cuda")

preprocess = inference_coco_transforms()
def prepare_frame(frame):
  tensor = preprocess(frame)
  batch_tensor = tensor.unsqueeze(0)
  batch_tensor = uniform_shape_masked_tensor(batch_tensor)
  return batch_tensor

red_mask = np.tile([0,0,255], reps=(height, width, 1)).astype(np.uint8)
green_mask = np.tile([0,255,0], reps=(height, width, 1)).astype(np.uint8)
blue_mask = np.tile([255,0,0], reps=(height, width, 1)).astype(np.uint8)
color_templates = [red_mask, green_mask, blue_mask]


def draw_mask(frame, predicted, alpha=0.5):
  frame = np.transpose(frame, (2, 0, 1)) # HWC -> CHW
  predicted_masks = predicted['segmentation_masks'].squeeze(0).cpu().numpy()
  
  results = []
  for frame_channel_slice, mask in zip(frame, predicted_masks):
    mask = (mask*255).astype(np.uint8)
    result = cv2.addWeighted(frame_channel_slice, 1-alpha, mask, alpha, 0.0)
    results.append(result)

  results = np.stack(results)
  results = np.transpose(results, (1, 2, 0))
  return results



for i in range(total_frame_number):
  torch.cuda.empty_cache()

  ok, frame = vc.read()
  count = count + 1

  inputs = prepare_frame(frame)

  with torch.no_grad():
    predicted = model(inputs.to(device))

  output_frame = draw_mask(frame, predicted, alpha=0.5)

  vw.write(output_frame)

  print("\r{}/{}, {:.2f} s, {:.1f} fps".format(count,
    total_frame_number,
    time.time() - start_time,
    count/(time.time() - start_time)), end="")

print()

vw.release()