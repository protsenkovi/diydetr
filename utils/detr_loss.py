
import torch
from utils.matcher import matcher
from torch.nn.functional import binary_cross_entropy
from utils.losses import diou_loss
from utils.functions import uniform_shape_masks
import torch.nn.functional as F 

def compute_loss(predicted, targets):
  bs, num_queries, num_classes = predicted['class_predictions'].shape
  device = predicted['class_predictions'].device
  label2tensor = torch.eye(num_classes, device=device) # move 

  matchings = matcher(predicted=predicted, targets=targets)

  # class loss 
  # eof is 111110?
  target_class_labels = [t['labels'] for t in targets]

  target_class_distributions = torch.zeros_like(predicted['class_predictions'], device=device)
  target_class_distributions[:,:,-1] = 1.0

  for image_idx, matching in enumerate(matchings):
    if matching is not None:
      i, j = matching
      target_class_distributions[image_idx,i,:] = label2tensor[target_class_labels[image_idx][j]]

  class_loss = binary_cross_entropy(predicted['class_predictions'], target_class_distributions).mean()

  # bbox loss
  target_bboxes_sparse = [t['boxes'] for t in targets]
  target_bboxes_dense = torch.zeros_like(predicted['bbox_predictions'], device=device)

  for image_idx, matching in enumerate(matchings):
    if matching is not None:
      i, j = matching
      target_bboxes_dense[image_idx,i,:] = target_bboxes_sparse[image_idx][j].to(device)

  bbox_loss = torch.cat([diou_loss(u,v) for u,v in zip(predicted['bbox_predictions'], target_bboxes_dense)]).mean()

  # segmentation loss
  target_segmentation_masks = uniform_shape_masks([t['masks'] for t in targets]).to(device=device)
  target_segmentation_masks_with_missing = torch.zeros_like(predicted['segmentation_masks'], device=device)

  for image_idx, matching in enumerate(matchings):
    if matching is not None:
      i, j = matching
      target_masks = target_segmentation_masks[image_idx][j].to(device)
      if target_masks.shape[1:] != target_segmentation_masks_with_missing[image_idx].shape[1:]:
        # print("mismatch! {} {}".format(target_masks.shape, target_segmentation_masks_with_missing[image_idx].shape))
        target_masks = target_masks.unsqueeze(1)
        target_masks = F.interpolate(target_masks, size=target_segmentation_masks_with_missing[image_idx].shape[1:])
        target_masks = target_masks.squeeze(1)

      target_segmentation_masks_with_missing[image_idx,i] = target_masks
  segmentation_loss = 0.0
  for predicted_image_masks, target_image_masks in zip(predicted['segmentation_masks'], target_segmentation_masks_with_missing):
    segmentation_loss += binary_cross_entropy(predicted_image_masks, target_image_masks).mean()
  segmentation_loss /= bs

  return class_loss + bbox_loss + segmentation_loss