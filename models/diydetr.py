import torch
from torch import nn
from utils.masked_tensor import MaskedTensor
from models.components.cnn_encoder import CNNEncoder
from models.components.transformer import Transformer
from models.components.mlp import MLP
from models.components.segmentation_head import SegmentationHead
from utils.functions import batch_positional_encoding
from utils import config 

class DIYDETR(nn.Module):
  def __init__(
      self, 
      embed_dim,
      num_classes,
      num_object_slots=100,
      num_transformer_layers=2,
      number_of_resnet_layers=18,
      num_heads=2,
      dropout=0.0,
      device=torch.device("cpu")
  ):
    super().__init__() 
    # assert embed_dim//num_heads >= 16
    self.num_classes = num_classes
    self.num_object_slots = num_object_slots
    self.embed_dim = embed_dim
    self.num_transformer_layers = num_transformer_layers
    self.dropout = dropout
    self.device = device

    self.cnnencoder = CNNEncoder( 
      embed_dim=embed_dim,
      number_of_resnet_layers=number_of_resnet_layers
    )
    self.transformer = Transformer(
        embed_dim=embed_dim, 
        num_encoder_layers=num_transformer_layers, 
        num_decoder_layers=num_transformer_layers,
        num_heads=num_heads,
        dropout=dropout
    )
    self.segmentation_head = SegmentationHead(
      embed_dim=embed_dim,
      num_object_slots=num_object_slots,
      backbone_intermediate_layers_num_channels=self.cnnencoder.resnet.intemediate_layers_num_channels,
      num_heads=num_heads,
      dropout=dropout
    )
    self.bbox_regression_head = nn.Sequential(
      MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=4, num_layers=3), 
      nn.Sigmoid()
    )
    self.classification_head = nn.Sequential(
      nn.Linear(embed_dim, num_classes), 
      nn.Softmax(dim=-1)
    )

    self.positional_encoding = None
    self.object_slots_embedding = None 
    self.object_slots_positional_encoding = None

    self.to(self.device)

  def __device(self):
    return next(self.parameters()).device

  def __init_query__(self, image_embeddings_shape): # watch for batch size and image embedding sequence length.
    if self.positional_encoding is None \
       or (self.positional_encoding is not None \
           and self.positional_encoding.shape != image_embeddings_shape):
      self.positional_encoding = batch_positional_encoding(batch_shape=image_embeddings_shape)
      self.positional_encoding = self.positional_encoding.to(self.__device())

    if self.object_slots_embedding is None:
      self.object_slots_embedding = torch.zeros(size=(image_embeddings_shape[0], self.num_object_slots, self.embed_dim))
      self.object_slots_embedding = self.object_slots_embedding.to(self.__device())

    if self.object_slots_positional_encoding is None:
      self.object_slots_positional_encoding = torch.normal(mean=0, std=1.0, size=(image_embeddings_shape[0], self.num_object_slots, self.embed_dim))
      self.object_slots_positional_encoding.requires_grad_(True)
      self.object_slots_positional_encoding = self.object_slots_positional_encoding.to(self.__device())


  def forward(self, images:MaskedTensor):
    cnn_output, interm_outputs = self.cnnencoder(images)
    
    image_embeddings = cnn_output.data
    padding_mask = cnn_output.mask.squeeze(-1) if cnn_output.mask is not None else None

    
    self.__init_query__(image_embeddings.shape)

    answer, memory = self.transformer(
        query = self.object_slots_embedding,
        query_positional_encoding = self.object_slots_positional_encoding,
        memory_data = image_embeddings,
        memory_padding_mask = padding_mask,
        memory_positional_encoding = self.positional_encoding
    )

    segmentation_masks = self.segmentation_head(
        answer = answer, 
        memory = memory,
        memory_padding_mask = padding_mask, 
        backbone_intermediate_layers_outputs = interm_outputs
    )

    class_predictions = self.classification_head(answer)

    bbox_predictions = self.bbox_regression_head(answer)

    output = {
      'class_predictions':class_predictions, 
      'bbox_predictions':bbox_predictions, 
      'segmentation_masks':segmentation_masks
    }
    return output