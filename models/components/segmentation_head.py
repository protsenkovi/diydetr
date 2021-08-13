import math 
import torch
from torch import nn
import torch.nn.functional as F
from models.components.mask_proposer import MaskProposer
import einops

class SegmentationHead(nn.Module):
  """
    Simple convolutional head, using group norm. 
    Upsampling is in the Feature Pyramids Network style.
  """

  def __init__(
      self, 
      embed_dim, 
      backbone_intermediate_layers_num_channels, 
      num_object_slots,
      num_heads=8,
      dropout=0.1):
    super().__init__()
    self.num_object_slots = num_object_slots

    self.mask_proposer = MaskProposer(
        embed_dim = embed_dim,
        num_heads = num_heads,
        dropout = dropout
    )

    ds = [
      embed_dim,
      embed_dim//2**1,
      embed_dim//2**2,
      embed_dim//2**3,
      embed_dim//2**4,
    ]

    self.initial_conv = nn.Sequential(
        nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
        nn.GroupNorm(2, embed_dim),
        nn.ReLU()
    )

    self.feature_pyramid_conv = []
    self.adapters = [] 
    for i in range(3):
      self.feature_pyramid_conv.append(
        nn.Sequential(
          nn.Conv2d(ds[i], ds[i+1], kernel_size=3, padding=1),
          nn.GroupNorm(1, ds[i+1]),
          nn.ReLU()
        )
      )
      self.adapters.append(
        nn.Conv2d(backbone_intermediate_layers_num_channels[i], ds[i+1], kernel_size=1)
      )
    self.feature_pyramid_conv = nn.ModuleList(self.feature_pyramid_conv)
    self.adapters = nn.ModuleList(self.adapters)

    self.last_feature_pyramid_conv = nn.Sequential(
      nn.Conv2d(ds[-2], ds[-1], kernel_size=3, padding=1),
      nn.GroupNorm(1, ds[-1]),
      nn.ReLU()
    )

    self.last_conv = nn.Conv2d(ds[-1], 1, kernel_size=3, padding=1)

  def forward(
      self,
      answer,
      memory, 
      memory_padding_mask,
      backbone_intermediate_layers_outputs
  ):
    segmentation_mask_proposal = self.mask_proposer(
        query=answer, 
        key=memory, 
        key_padding_mask=memory_padding_mask, 
        backbone_embedding_shape=backbone_intermediate_layers_outputs['layer5'].shape
    )

    x = self.initial_conv(segmentation_mask_proposal)

    for i in range(3):
      x = self.feature_pyramid_conv[i](x)
      fpn = self.adapters[i](backbone_intermediate_layers_outputs['layer{}'.format(3-i)].data) 
      fpn = einops.repeat(fpn, "b c h w -> b masks_count c h w", masks_count=self.num_object_slots)
      fpn = einops.rearrange(fpn, "b masks_count c h w -> (b masks_count) c h w")
      x = fpn + F.interpolate(x, size=fpn.shape[-2:], mode="nearest")

    x = self.last_feature_pyramid_conv(x)
    x = self.last_conv(x)
    x = torch.sigmoid(x)
    x = F.interpolate(x, size=(fpn.shape[-2]*4, fpn.shape[-1]*4), mode="nearest")
    x = einops.rearrange(x, "(b masks_count) c h w -> b masks_count c h w", masks_count=self.num_object_slots)

    x_mask = F.interpolate(backbone_intermediate_layers_outputs['layer{}'.format(3-i)].mask.float(),
                           size=(fpn.shape[-2]*4, fpn.shape[-1]*4), mode="nearest").bool()
    x_mask = einops.repeat(x_mask, "b c h w -> b masks_count c h w", masks_count=self.num_object_slots)

    x = x.masked_fill(x_mask, 0.0)
    x = x.squeeze(2)

    return x