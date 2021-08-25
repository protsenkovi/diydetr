import torchvision
from datasets.transformations import ConvertCocoPolysToMask, Compose, ToTensor, Normalize, RandomHorizontalFlip, \
RandomSelect, RandomResize, RandomSizeCrop
import torchvision.transforms.functional as F

class CocoDetection(torchvision.datasets.CocoDetection):
  def __init__(
      self,
      img_folder,
      ann_file,
      transforms,
      return_masks):
    super(CocoDetection, self).__init__(img_folder, ann_file)
    self._transforms = transforms
    self.prepare = ConvertCocoPolysToMask(return_masks)

  def __getitem__(self, idx):
    img, target = super(CocoDetection, self).__getitem__(idx)
    image_id = self.ids[idx]
    target = {'image_id': image_id, 'annotations': target}
    img, target = self.prepare(img, target)
    if self._transforms is not None:
      img, target = self._transforms(img, target)
    return img, target

def train_coco_transforms():
  """ image set \in {"train", "val"}. Where is test? """
  normalize = Compose([
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  size_choices = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

  return Compose([
    RandomHorizontalFlip(),
    RandomSelect(
      RandomResize(size_choices, max_size=1333),
      Compose([
        RandomResize(size_choices=[400, 500, 600]),
        RandomSizeCrop(384, 600),
        RandomResize(size_choices=size_choices, max_size=1333)
      ])
    ),
    normalize,
  ])

def val_coco_transforms():
  normalize = Compose([
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  size_choices = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

  return Compose([
    RandomResize(size_choices=[800], max_size=1333),
    normalize
  ])

class Preprocess(object): # use F.normalize, box_xyxy_cxcywh, torch.tensor
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std
  
  def __call__(self, image):
    image = F.to_tensor(image)
    image = F.normalize(image, mean=self.mean, std=self.std)
    return image

def inference_coco_transforms():
  return Preprocess(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
