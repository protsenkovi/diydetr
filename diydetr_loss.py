import torch
from utils.matcher import matcher
from torch.nn.functional import binary_cross_entropy
from utils.losses import elementwise_strange_loss
from utils.functions import box_cxcywh_to_xyxy, uniform_shape_masks
import torch.nn.functional as F 
from utils import config

def diydetr_loss(predicted, targets):
  """ Matcher uses pairwise distance. Loss uses element-wise distance. 
      Predicted format is cx cy w h, targets format is x y x y.
  """
  predicted['bbox_predictions'] = box_cxcywh_to_xyxy(predicted['bbox_predictions'])

  bs, num_queries, num_classes = predicted['class_predictions'].shape
  device = predicted['class_predictions'].device
   # move 

  matchings = matcher(predicted=predicted, targets=targets)
  
  # class loss 
  target_class_ids = [t['ids'] for t in targets]
  target_class_distributions = torch.zeros_like(predicted['class_predictions'], device=device)
  target_class_distributions[:,:,-1] = 1.0

  for image_idx, matching in enumerate(matchings):
    if matching is not None:
      i, j = matching
      target_class_distributions[image_idx,i,:] = config.label2tensor[target_class_ids[image_idx][j]].to(device)

  class_loss = binary_cross_entropy(predicted['class_predictions'], target_class_distributions).mean()


  # bbox loss
  target_bboxes_sparse = [t['boxes'] for t in targets]
  target_bboxes_dense = torch.zeros_like(predicted['bbox_predictions'], device=device)
  for image_idx, matching in enumerate(matchings):
    if matching is not None:
      i, j = matching
      target_bboxes_dense[image_idx,i,:] = target_bboxes_sparse[image_idx][j].to(device)

  bbox_loss = torch.cat([elementwise_strange_loss(u,v) for u,v in zip(predicted['bbox_predictions'], target_bboxes_dense)]).mean()

  # segmentation loss
  target_segmentation_masks = uniform_shape_masks([t['masks'] for t in targets]).to(device=device)
  target_segmentation_masks_with_missing = torch.zeros_like(predicted['segmentation_masks'], device=device)

  for image_idx, matching in enumerate(matchings):
    if matching is not None:
      i, j = matching
      target_masks = target_segmentation_masks[image_idx][j].to(device)
      if target_masks.shape[1:] != target_segmentation_masks_with_missing[image_idx].shape[1:]:
        target_masks = target_masks.unsqueeze(1)
        target_masks = F.interpolate(target_masks, size=target_segmentation_masks_with_missing[image_idx].shape[1:])
        target_masks = target_masks.squeeze(1)

      target_segmentation_masks_with_missing[image_idx,i] = target_masks
  segmentation_loss = 0.0
  for predicted_image_masks, target_image_masks in zip(predicted['segmentation_masks'], target_segmentation_masks_with_missing):
    segmentation_loss += binary_cross_entropy(predicted_image_masks, target_image_masks).mean()
  segmentation_loss /= bs

  config.tb.add_scalar("loss/class", class_loss, config.epoch)
  config.tb.add_scalar("loss/bbox", bbox_loss, config.epoch)
  config.tb.add_scalar("loss/segmentation", segmentation_loss, config.epoch)

  return class_loss + bbox_loss + segmentation_loss