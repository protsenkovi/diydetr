import torch
import einops
from torch import Tensor
from typing import Optional


class MaskedTensor(object):
  def __init__(self, data:Tensor, mask:Optional[Tensor]=None):
    self.data = data
    self.shape = data.shape
    self.mask = mask
    
  def to(self, device):
    return MaskedTensor(self.data.to(device), self.mask.to(device) if self.mask is not None else None)

  def __repr__(self):
    return "MaskedTensor shapes {} {}".format(self.data.shape, self.mask.shape if self.mask is not None else None)

  def __delegate_op(self, op, *args, **kargs):
    data = op(self.data, *args, **kargs)
    if self.mask is not None:
      mask = op(self.mask, *args, **kargs)
    return MaskedTensor(data, mask)

  def rearrange(self, *args, **kargs):
    return self.__delegate_op(einops.rearrange, *args, **kargs)

  def pin_memory(self):
    self.data = self.data.pin_memory()
    if self.mask is not None:
      self.mask = self.mask.pin_memory()
    return self