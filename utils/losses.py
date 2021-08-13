import torch
from utils.functions import box_xyxy_to_cxcy


def intersection_and_union(boxes1, boxes2):
  """ 
      cross-intersection and cross-union.
      Expected format of the box left-top, right-bottom coordinates.
  """
  area1 = torch.prod(boxes1[:, 2:] - boxes1[:, :2], dim=-1)
  area2 = torch.prod(boxes2[:, 2:] - boxes2[:, :2], dim=-1)

  # non-trivial operation
  intersection_lt = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # maximum
  intersection_rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

  wh = (intersection_rb - intersection_lt).clamp(min=0)
  intersection = wh[:, :, 0] * wh[:, :, 1]
  union = area1[:, None] + area2 - intersection

  return intersection, union

def giou(boxes1, boxes2):
  """ https://giou.stanford.edu/ """
  assert (boxes1[:, :2] <= boxes1[:, 2:] ).all()
  assert (boxes2[:, :2] <= boxes2[:, 2:] ).all()
  intersection, union = intersection_and_union(boxes1, boxes2)
  
  enclosure_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
  enclosure_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

  wh = (enclosure_rb - enclosure_lt).clamp(min=0)
  enclosure = wh[:, :, 0] * wh[:, :, 1]

  return intersection/union + union/enclosure



def diou(boxes1, boxes2):
  intersection, union = intersection_and_union(boxes1, boxes2)
  
  enclosure_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
  enclosure_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

  wh = (enclosure_rb - enclosure_lt).clamp(min=0)
  enclosure_diagonal = torch.norm(enclosure_lt-enclosure_rb, dim=-1, p=2)
  enclosure = wh[:, :, 0] * wh[:, :, 1]

  distance_between_box_centers = torch.cdist(
      box_xyxy_to_cxcy(boxes1),
      box_xyxy_to_cxcy(boxes2), 
      p=2
  )

  return intersection/union + union/enclosure - distance_between_box_centers/enclosure_diagonal

def diou_loss(boxes1, boxes2):
  """
  Minimization objectives:
    - difference between intersection area and union area,
    - difference between enclosure area and union area (for gradient of non-overlapped boxes),
    - distance between centers (helping iou, for convergence speed).
    
  """
  diou_val = diou(boxes1, boxes2)
  return 2 - diou_val
