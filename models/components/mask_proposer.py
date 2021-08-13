import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional
import einops
import math

class MaskProposer(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(
        self, 
        embed_dim,
        num_heads = 2,
        dropout = 0.0
    ):
      super().__init__()
      self.num_heads = num_heads
      self.embed_dim = embed_dim

      self.dropout = nn.Dropout(dropout)
      self.q_linear = nn.Linear(embed_dim, embed_dim)
      self.k_linear = nn.Linear(embed_dim, embed_dim) 

      self._reset_parameters()

    def _reset_parameters(self):
      for p in self.parameters():
          if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    def forward(
        self, 
        query:Tensor, 
        key:Tensor,
        key_padding_mask:Optional[Tensor],
        backbone_embedding_shape
    ):
      
      if key_padding_mask is not None:
        key_padding_mask = einops.repeat(
            key_padding_mask, 
            "bs key_len -> (bs heads) query_len key_len", 
            heads = self.num_heads,
            query_len = query.shape[1])

      q = self.q_linear(query)
      k = self.k_linear(key)

      q = einops.rearrange(q, "bs n (heads d) -> (bs heads) n d", heads=self.num_heads)
      k = einops.rearrange(k, "bs n (heads d) -> (bs heads) n d", heads=self.num_heads)

      q = q / math.sqrt(self.embed_dim / self.num_heads)
      attn = torch.bmm(q, k.transpose(-2, -1))

      # set the lowest possible energy to masked keys
      if key_padding_mask is not None:
        attn.masked_fill_(key_padding_mask, float("-inf"))

      attn = F.softmax(attn, dim=-1) 

      attn = self.dropout(attn)
      attn = einops.rearrange(
          attn, 
          "(bs heads) query_len key_len -> bs heads query_len key_len", 
          heads=self.num_heads
      ) 
      attn = attn.mean(dim=1)

      proposal = attn
      proposal = einops.rearrange(
          proposal,
          "b mask_count (h w) -> b mask_count h w",
          h = backbone_embedding_shape[-2],
          w = backbone_embedding_shape[-1]
      )
      proposal = einops.repeat(proposal, "b mask_count h w -> b c mask_count h w", c=backbone_embedding_shape[-3])
      proposal = einops.rearrange(proposal, "b c mask_count h w -> (b mask_count) c h w")
      return proposal