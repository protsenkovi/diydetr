import copy
import torch
from utils.masked_tensor import MaskedTensor

def box_xyxy_to_cxcywh(x):
  x0,y0,x1,y1 = x.unbind(-1)
  b = [
       (x0+x1)/2, (y0+y1)/2,
       (x1-x0), (y1-y0)
  ]
  return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
  cx,cy,w,h = x.unbind(-1)
  b = [
       cx-w/2, cy-h/2,
       cx+w/2, cy+h/2
  ]
  return torch.stack(b, dim=-1)

def box_xyxy_to_cxcy(x):
  x0,y0,x1,y1 = x.unbind(-1)
  b = [
       (x0+x1)/2, (y0+y1)/2
  ]
  return torch.stack(b, dim=-1)

def uniform_shape_masked_tensor(tensors):
  """ 
    expect images in c h w format.
    Via padding to the right converts list of non-uniform shaped tensors 
    to one tensor.
    Returns object of type MaskedTensor that contain data and mask 
    with extra prefix dimension. """
  max_shape, _ = torch.tensor([tensor.shape for tensor in tensors]).max(dim=0)
  c, h, w = max_shape
  batch = torch.zeros(size=(len(tensors), c, h, w))
  mask = torch.full(fill_value=True, size=(len(tensors), 1, h, w))
  for i, tensor in enumerate(tensors):
    batch[i, :tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor
    mask[i, :tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = False
  return MaskedTensor(batch, mask)

def uniform_shape_masks(tensors):
  """ 
    expect images in c h w format.
    Via padding to the right converts list of non-uniform shaped tensors 
    to one tensor.
    Returns object of type MaskedTensor that contain data and mask 
    with extra prefix dimension. """
  max_shape, _ = torch.tensor([tensor.shape for tensor in tensors]).max(dim=0)
  count, h, w = max_shape
  batch = torch.zeros(size=(len(tensors), count, h, w))
  for i, tensor in enumerate(tensors):
    batch[i, :tensor.shape[0], :tensor.shape[1], :tensor.shape[2]] = tensor
  return batch

def get_clones(module, n, exclude=[]):
  memo = {id(obj):obj for obj in exclude}
  return [copy.deepcopy(module, memo) for i in range(n)]

def batch_positional_encoding(batch_shape, temperature = 10000):
  bs, l, d = batch_shape
  t = torch.arange(d, dtype=torch.float32, device=torch.device("cpu"))
  pos = torch.arange(l, dtype=torch.float32, device=torch.device("cpu"))
  denom = temperature ** (2 * t / d)
  x = pos.unsqueeze(1) / denom.unsqueeze(0)
  encoding = torch.stack([x[:,0::2].sin(), x[:,1::2].cos()], dim=-1).flatten(-2)
  encoding = encoding.unsqueeze(0)
  encoding = torch.broadcast_to(encoding, (bs, l, d))
  return encoding