from torch import nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from utils.masked_tensor import MaskedTensor
import torch.nn.functional as F

class ResNet(nn.Module):
  def __init__(
      self,
      number_of_layers,
      train_backbone:bool,
      return_intermediate_layers:bool,
      replace_stride_with_dilation:bool
  ):
    super().__init__()

    assert number_of_layers in [18,34,50,101,152]

    name = 'resnet{}'.format(number_of_layers)

    num_channels = 512 if name in ['resnet18', 'resnet34'] else 2048 

    backbone = getattr(torchvision.models, name) (
        replace_stride_with_dilation=[False, False, replace_stride_with_dilation],
        pretrained=True,
        norm_layer=torchvision.ops.misc.FrozenBatchNorm2d
    )

    if train_backbone: 
      trainable_layer_ids = [2,3,4]
      trainable_layers_names = ['layer{}'.format(i) for i in trainable_layer_ids]
      for name, parameter in backbone.named_parameters():
        if any([l in name for l in trainable_layers_names]): 
          # TODO Improve naming. Simplify condition. Should be layer_name in trainable_layer_names
          parameter.requires_grad_(True)
        else:
          parameter.requires_grad_(False)
    else:
      for name, parameter in backbone.named_parameters():
        parameter.requires_grad_(False)

    if return_intermediate_layers:
      return_layer_ids = [1,2,3,4] 
    else:
      return_layer_ids = [4]

    self.return_layers = {'layer{}'.format(name):i for i,name in enumerate(return_layer_ids)}

    self.body = IntermediateLayerGetter(backbone, return_layers=self.return_layers)
    self.num_channels = num_channels
    self.intemediate_layers_num_channels = [num_channels//2**(i+1) for i in range(3)]
    
  def forward(
      self,
      batch: MaskedTensor
  ):
    xs = self.body(batch.data)

    output = {}
    for name, (_, x) in zip(self.return_layers, xs.items()):
      if batch.mask is not None:
        mask = F.interpolate(batch.mask.float(), size=x.shape[2:]).bool()
      else:
        mask = None
      output[name] = MaskedTensor(
          x, 
          mask 
      )
    
    return output
