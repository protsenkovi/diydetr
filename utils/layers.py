from torch import nn
from utils.masked_tensor import MaskedTensor

class Sequential(nn.Sequential):
  def forward(self, input, **kwargs):
    for module in self:
      input = module(input, **kwargs)
    return input

class Dropout(nn.Dropout):
  def forward(self, x:MaskedTensor):
    return MaskedTensor(super().forward(x.data), x.mask)

class LayerNorm(nn.LayerNorm):
  def forward(self, x:MaskedTensor):
    return MaskedTensor(super().forward(x.data), x.mask)

class ReLU(nn.ReLU):
  def forward(self, x:MaskedTensor):
    return MaskedTensor(super().forward(x.data), x.mask)

class Linear(nn.Linear):
  def forward(self, x:MaskedTensor):
    res = super().forward(x.data)
    return MaskedTensor(res, x.mask)

class Conv2d(nn.Conv2d):
  def forward(self, x:MaskedTensor):
    return MaskedTensor(super().forward(x.data), x.mask)


    