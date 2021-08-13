import torch
from torch import nn
import torchvision
from utils.masked_tensor import MaskedTensor
from models.components.resnet import ResNet


class CNNEncoder(nn.Module):
  def __init__(
      self,
      embed_dim,
      number_of_resnet_layers = 18,
      train_backbone = True,
      return_intermediate_layers = True,
      replace_stride_with_dilation = False
  ):
    super().__init__()

    self.resnet = ResNet(
      number_of_layers = number_of_resnet_layers,
      train_backbone = train_backbone,
      return_intermediate_layers = return_intermediate_layers,
      replace_stride_with_dilation = replace_stride_with_dilation
    ) 

    self.proj = nn.Conv2d(
        in_channels = self.resnet.num_channels,
        out_channels = embed_dim,
        kernel_size=(1,1)
    )

  def forward(self, x:MaskedTensor):
    xs = self.resnet(x)
    xs['layer5'] = MaskedTensor(self.proj(xs['layer4'].data), xs['layer4'].mask)
    x = xs['layer5'].rearrange("b c h w -> b (h w) c")
    return x, xs