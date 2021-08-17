import torch
from utils.functions import box_xyxy_to_cxcy, box_xyxy_center
from utils import config

def elementwise_intersection_and_union(boxes1, boxes2):
  """ 
      elementwise-intersection and elementwise-union.
      Expected format of the box left-top, right-bottom coordinates.
  """
  area1 = torch.prod(boxes1[:, 2:] - boxes1[:, :2], dim=-1)
  area2 = torch.prod(boxes2[:, 2:] - boxes2[:, :2], dim=-1)

  intersection_lt = torch.max(boxes1[:, :2], boxes2[:, :2]) 
  intersection_rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])

  wh = (intersection_rb - intersection_lt).clamp(min=0)
  intersection = wh[:, 0] * wh[:, 1]
  union = area1 + area2 - intersection

  return intersection, union

def pairwise_intersection_and_union(boxes1, boxes2):
  """ 
      pairwise intersection and union.
      Expected format of the box left-top, right-bottom coordinates.
  """
  area1 = torch.prod(boxes1[:, 2:] - boxes1[:, :2], dim=-1)
  area2 = torch.prod(boxes2[:, 2:] - boxes2[:, :2], dim=-1)

  intersection_lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) 
  intersection_rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

  wh = (intersection_rb - intersection_lt).clamp(min=0)
  intersection = wh[:, :, 0] * wh[:, :, 1]
  union = area1 + area2 - intersection

  return intersection, union


def elementwise_strange_loss(boxes1, boxes2):
  """ Input format xyxy """

  distance_between_box_centers = torch.norm(
      box_xyxy_center(boxes1) - box_xyxy_center(boxes2),
      p=2,
      dim=1
  )

  boxes_shift = box_xyxy_center(boxes2) - box_xyxy_center(boxes1)
  boxes2 = boxes2.clone()
  boxes2 -= torch.cat([boxes_shift,boxes_shift],dim=1)

  intersection, union = elementwise_intersection_and_union(boxes1, boxes2)

  iou, d =  intersection/union, distance_between_box_centers
  return (1-iou+0.1)*(d + 0.1)-0.01

def pairwise_strange_loss(boxes1, boxes2):
  """ Input format xyxy """

  distance_between_box_centers = torch.cdist(
      box_xyxy_center(boxes1), 
      box_xyxy_center(boxes2),
      p=2
  )

  boxes_shift = box_xyxy_center(boxes2).unsqueeze(0) - box_xyxy_center(boxes1).unsqueeze(1)
  boxes_shift = torch.cat([boxes_shift, boxes_shift], dim=2)

  boxes2_aligned = boxes2.clone().repeat(boxes1.shape[0], 1).view(boxes1.shape[0], boxes2.shape[0], 4)
  boxes2_aligned -= boxes_shift

  area1 = torch.prod(boxes1[:, 2:] - boxes1[:, :2], dim=-1)
  area2 = torch.prod(boxes2[:, 2:] - boxes2[:, :2], dim=-1)

  intersection_lt = torch.max(boxes1[:, None, :2].repeat(1,boxes2.shape[0],1), boxes2_aligned[:, :, :2]) 
  intersection_rb = torch.min(boxes1[:, None, 2:].repeat(1,boxes2.shape[0],1), boxes2_aligned[:, :, 2:])
  intersection_boxes = torch.cat([intersection_lt,intersection_rb], dim=-1)

  wh = (intersection_rb - intersection_lt).clamp(min=0)
  intersection = wh[:, :, 0] * wh[:, :, 1]
  union = area1[:, None].repeat(1,area2.shape[0]) + area2[None,:].repeat(area1.shape[0],1) - intersection

  iou, d =  intersection/union, distance_between_box_centers
  return (1-iou+0.1)*(d + 0.1)-0.01