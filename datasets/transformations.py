import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as F_torch

# from torch.nn.functional import interpolate
from torchvision.ops.misc import interpolate
import PIL
from pycocotools import mask as coco_mask
import random
from utils.functions import box_xyxy_to_cxcywh
from utils import config

def crop(image, target, region):
  cropped_image = F.crop(image, *region)

  target = target.copy()
  i,j,h,w = region

  target["size"] = torch.tensor([h, w])

  fields = ["labels", "area", "iscrowd"]

  if "boxes" in target:
    boxes = target["boxes"]
    max_size = torch.as_tensor([w, h], dtype=torch.float32)
    cropped_boxes = boxes - torch.as_tensor([j,i,j,i])
    cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
    cropped_boxes = cropped_boxes.clamp(min=0)
    area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
    target["boxes"] = cropped_boxes.reshape(-1, 4)
    fields.append("boxes")

  if "masks" in target:
    target['masks'] = target['masks'][:, i:i+h, j:j+w]
    fields.append("masks")

  if "boxes" in target or "masks" in target:
    if "boxes" in target:
      cropped_boxes = target['boxes'].reshape(-1, 2, 2)
      keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
    else:
      keep = target['masks'].flatten(1).any(1)

    for field in fields:
      target[field] = target[field][keep]

  return cropped_image, target

def hflip(image, target):
  flipped_image = F.hflip(image)

  w, h = image.size

  target = target.copy()
  if "boxes" in target:
    boxes = target["boxes"]
    boxes = boxes[:, [2,1,0,3]] * torch.as_tensor([-1,1,-1,1]) + torch.as_tensor([w,0,w,0])
    target["boxes"] = boxes

  if "masks" in target:
    target["masks"] = target["masks"].flip(-1)

  return flipped_image, target

def resize(image, target, size, max_size=None):
  """ size can be min_size or (w,h) tuple. 
      complicated method with lots of edge cases.
  """
  def get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
      min_original_size = float(min(w,h))
      max_original_size = float(max(w,h))
      if max_original_size / min_original_size * size > max_size:
        size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
      return (h, w)

    if w < h:
      ow = size
      oh = int(size * h / w)
    else:
      oh = size
      ow = int(size * w / h)

    return (oh, ow)

  def get_size(image_size, size, max_size=None):
    if isinstance(size, (list, tuple)):
      return size[::-1]
    else:
      return get_size_with_aspect_ratio(image_size, size, max_size)

  size = get_size(image.size, size, max_size)
  rescaled_image = F.resize(image, size)

  if target is None:
    return rescaled_image, None

  ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
  ratio_width, ratio_height = ratios

  target = target.copy()
  if "boxes" in target:
    boxes = target["boxes"]
    scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    target["boxes"] = scaled_boxes

  if "area" in target:
    area = target["area"]
    scaled_area = area * (ratio_width * ratio_height)
    target["area"] = scaled_area

  h, w = size
  target["size"] = torch.tensor([h, w])

  if "masks" in target:
    target['masks'] = interpolate(
        target['masks'][:, None].float(), size, mode="nearest"
    )[:, 0] > 0.5

  return rescaled_image, target


class RandomSizeCrop(object): # use T.RandomCrop.getParams and and crop
  def __init__(self, min_size, max_size):
    self.min_size = min_size
    self.max_size = max_size

  def __call__(self, img: PIL.Image.Image, target:dict):
    w = random.randint(self.min_size, min(img.width, self.max_size))
    h = random.randint(self.min_size, min(img.height, self.max_size))
    region = T.RandomCrop.get_params(img, [h, w])
    return crop(img, target, region)


class RandomHorizontalFlip(object): # use hflip
  def __init__(self, p=0.5):
    self.p = p 

  def __call__(self, img, target):
    if random.random() < self.p:
      return hflip(img, target)
    return img, target

class RandomResize(object): # use resize
  def __init__(self, size_choices, max_size=None):
    assert isinstance(size_choices, (list, tuple))
    self.size_choices = size_choices
    self.max_size = max_size

  def __call__(self, img, target=None):
    size = random.choice(self.size_choices)
    return resize(img, target, size, self.max_size)

class RandomSelect(object): # randomly selects between transforms1 and transforms2
  def __init__(self,  transforms1, transforms2, p=0.5):
    self.transforms1 = transforms1
    self.transforms2 = transforms2
    self.p = p

  def __call__(self, img, target):
    if random.random() < self.p:
      return self.transforms1(img, target)
    return self.transforms2(img, target)

class ToTensor(object):
  def __call__(self, img, target):
    return F.to_tensor(img), target

class Normalize(object): # use F.normalize, box_xyxy_cxcywh, torch.tensor
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
  
  def __call__(self, image, target=None):
    image = F.normalize(image, mean=self.mean, std=self.std)
    if target is None:
      return image, None
    target = target.copy()
    h, w = image.shape[-2:]
    if "boxes" in target:
      boxes = target["boxes"]
      # boxes = box_xyxy_to_cxcywh(boxes)
      boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
      target["boxes"] = boxes
    if "labels" in target:
      target["ids"] = torch.tensor([config.class_labels_to_ids[label.item()] for label in target["labels"]])
    return image, target

class Compose(object):
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, image, target):
    for t in self.transforms:
      image, target = t(image, target)
    return image, target

def convert_coco_poly_to_mask(segmentation, height, width):
  masks = []
  for polygons in segmentation:
    rles = coco_mask.frPyObjects(polygons, height, width)
    mask = coco_mask.decode(rles)
    if len(mask.shape) < 3:
      mask = mask[..., None]
    mask = torch.as_tensor(mask, dtype=torch.uint8)
    mask = mask.any(dim=2)
    masks.append(mask)

  if masks:
    masks = torch.stack(masks, dim=0)
  else:
    masks = torch.zeros((0, height, width), dtype=torch.uint8)

  return masks

# bad name for class
class ConvertCocoPolysToMask(object):
  def __init__(self, return_masks=False):
    self.return_masks = return_masks

  def __call__(self, image, target):
    w, h = image.size

    image_id = target["image_id"]
    image_id = torch.tensor([image_id])

    annotations = target["annotations"]

    annotations = [obj for obj in annotations if 'iscrowd' not in obj or obj['iscrowd'] == 0]

    boxes = [obj["bbox"] for obj in annotations]
    # guard against boxes vie resizing (?)
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)

    classes = [obj["category_id"] for obj in annotations]
    classes = torch.tensor(classes, dtype=torch.int64)

    if self.return_masks:
      segmentations = [obj["segmentation"] for obj in annotations]
      masks = convert_coco_poly_to_mask(segmentations, h, w)

    keypoints = None
    if annotations and "keypoints" in annotations[0]:
      keypoints = [obj["keypoints"] for obj in annotations]
      keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
      num_keypoints = keypoints.shape[0]
      if num_keypoints:
        keypoints = keypoints.view(num_keypoints, -1, 3)

    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]
    if self.return_masks:
      masks = masks[keep]
    if keypoints is not None:
      keypoints = keypoints[keep]

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    if self.return_masks:
      target["masks"] = masks
    target["image_id"] = image_id
    if keypoints is not None:
      target["keypoints"] = keypoints

    area = torch.tensor([obj["area"] for obj in annotations])
    iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in annotations])
    target["area"] = area[keep]
    target["iscrowd"] = iscrowd[keep]

    target["orig_size"] = torch.as_tensor([int(h), int(w)])
    target["size"] = torch.as_tensor([int(h), int(w)])

    return image, target
  