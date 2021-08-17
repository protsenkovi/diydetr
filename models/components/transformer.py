import torch
from torch import nn
from torch import Tensor

from utils.functions import get_clones, batch_positional_encoding
from utils.layers import Sequential

from utils import config

class EncoderLayer(nn.Module):
  def __init__(
      self,
      embed_dim,
      num_heads,
      feedforward_expansion_coefficient,
      dropout,
      activation
    ):
    super().__init__()

    self.attention = nn.MultiheadAttention(
      embed_dim = embed_dim,
      num_heads = num_heads,
      dropout = dropout,
      batch_first = True
    )

    self.feedforward = nn.Sequential(
      nn.Linear(
        in_features = embed_dim,
        out_features = int(embed_dim*feedforward_expansion_coefficient)
      ),
      activation,
      nn.Dropout(dropout),
      nn.Linear(
        in_features = int(embed_dim*feedforward_expansion_coefficient),
        out_features = embed_dim
      )
    )

    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)

  def forward(self, x, positional_encoding, key_padding_mask, idx):
    
    residual, avg_head_attention_matrix = self.attention(
        query = x + positional_encoding, 
        key = x + positional_encoding, 
        value = x,
        key_padding_mask = key_padding_mask
    )
    x = self.norm1(x + self.drop1(residual))

    # print(idx, id(self.norm1))
    # print(avg_head_attention_matrix.shape)
    # print(avg_head_attention_matrix)
    # config.tb.add_images(self._get_name() + str(idx) + "inputs", 
    #   x.unsqueeze(1), config.epoch)

    # config.tb.add_scalar(self._get_name() + str(idx) + "residual_nan", residual.isnan().sum(), config.epoch)
    # config.tb.add_scalar(self._get_name() + str(idx) + "residual_inf", residual.isinf().sum(), config.epoch)
    # config.tb.add_scalar(self._get_name() + str(idx) + "x_nan", x.isnan().sum(), config.epoch)
    # config.tb.add_scalar(self._get_name() + str(idx) + "x_inf", x.isinf().sum(), config.epoch)

    residual = self.feedforward(x)
    x = self.norm2(x + self.drop2(residual))

    # config.tb.add_scalar(self._get_name() + str(idx) + "residual_nan", residual.isnan().sum(), config.epoch)
    # config.tb.add_scalar(self._get_name() + str(idx) + "residual_inf", residual.isinf().sum(), config.epoch)
    # config.tb.add_scalar(self._get_name() + str(idx) + "x_nan", x.isnan().sum(), config.epoch)
    # config.tb.add_scalar(self._get_name() + str(idx) + "x_inf", x.isinf().sum(), config.epoch)

    
    return x

class DecoderLayer(nn.Module):
  def __init__(
      self,
      embed_dim,
      num_heads,
      feedforward_expansion_coefficient,
      dropout,
      activation
  ):
    super().__init__()

    self.self_attention = nn.MultiheadAttention(
        embed_dim = embed_dim,
        num_heads = num_heads,
        dropout = dropout,
        batch_first = True
        
    )

    self.cross_attention = nn.MultiheadAttention(
        embed_dim = embed_dim,
        num_heads = num_heads,
        dropout = dropout,
        batch_first = True
    )

    self.feedforward = nn.Sequential(
      nn.Linear(
        in_features = embed_dim,
        out_features = int(embed_dim*feedforward_expansion_coefficient)
      ),
      activation,
      nn.Dropout(dropout),
      nn.Linear(
        in_features = int(embed_dim*feedforward_expansion_coefficient),
        out_features = embed_dim
      )
    )

    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)
    self.norm3 = nn.LayerNorm(embed_dim)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)
    self.drop3 = nn.Dropout(dropout)
 
  def forward(self, x, memory, query_positional_encoding, key_positional_encoding, key_padding_mask, idx):
    residual, avg_head_attention_matrix = self.self_attention(
      query = x + query_positional_encoding,
      key = x + query_positional_encoding,
      value = x
    )
    x = self.norm1(x + self.drop1(residual))

    residual, avg_head_attention_matrix = self.cross_attention(
      query = x + query_positional_encoding,
      key = memory + key_positional_encoding,
      value = memory,
      key_padding_mask = key_padding_mask
    )
    x = self.norm2(x + self.drop2(residual))

    residual = self.feedforward(x)
    x = self.norm3(x + self.drop3(residual))

    return x

class Sequential(nn.Sequential):
  def forward(self, input, **kwargs):
    for idx, module in enumerate(self):
      input = module(input, idx=idx, **kwargs)
      # config.tb.add_scalar(module._get_name() + str(idx) + "_nan", input.isnan().sum(), config.epoch)
      # config.tb.add_scalar(module._get_name() + str(idx) + "_inf", input.isinf().sum(), config.epoch)
    return input

class Transformer(nn.Module):
  """
  Batch dimension is first.
  """
  def __init__(
    self,
    embed_dim,
    num_heads=4,
    num_encoder_layers=3,
    num_decoder_layers=3,
    num_object_slots=100,
    feedforward_expansion_coefficient=4,
    dropout=0.1, #?
    activation=nn.ReLU()
  ):
    super(Transformer, self).__init__()
    self.num_object_slots = num_object_slots

    encoder_layer = EncoderLayer(
      embed_dim = embed_dim,
      num_heads = num_heads,
      feedforward_expansion_coefficient = feedforward_expansion_coefficient,
      dropout = dropout,
      activation = activation
    )
    self.encoder = Sequential(*get_clones(encoder_layer, num_encoder_layers))

    decoder_layer = DecoderLayer(
      embed_dim = embed_dim,
      num_heads = num_heads,
      feedforward_expansion_coefficient = feedforward_expansion_coefficient,
      dropout = dropout,
      activation = activation
    )
    self.decoder = Sequential(*get_clones(decoder_layer, num_decoder_layers))

    self._reset_parameters()

  def _reset_parameters(self):
    for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

  def forward(
    self, 
    query:Tensor,
    query_positional_encoding:Tensor,
    memory_data:Tensor,
    memory_padding_mask:Tensor,
    memory_positional_encoding:Tensor
  ):

    memory = self.encoder(
      memory_data,
      positional_encoding = memory_positional_encoding,
      key_padding_mask = memory_padding_mask
    )

    answer = self.decoder(
        query,
        memory = memory,
        query_positional_encoding = query_positional_encoding,
        key_positional_encoding = memory_positional_encoding,
        key_padding_mask = memory_padding_mask
    )

    return answer, memory