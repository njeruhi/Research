import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import einops

# Let us see it properly in action
class PatchEmbedding(torch.nn.Module):
    def __init__(self, in_channels=3, patch_size=8, embed_dim=128):
        super().__init__()
        self.proj = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, embed_dim, 4, 4 ) if patch_size=8
        x = self.proj(x)
        x = x.flatten(2) #(B, embed_dim, 4, 4) -> (B, embed_dim, 4x4)
        return x.transpose(1, 2) # (B, embed_dim, 16) -> (B, N, embed_dim) where N = H.W / patch_size**2
    
class EncoderLayer(nn.Module):
  def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
    super(EncoderLayer, self).__init__()
    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)
    self.layer_norm1 = nn.LayerNorm(d_model)
    self.layer_norm2 = nn.LayerNorm(d_model)

  def forward(self, x):
    # Self-attention layer
    attn_output, _ = self.self_attn(x, x, x, attn_mask=None)
    x = self.layer_norm1(x + attn_output)  # Add & Norm

    # Feed Forward layer
    x = self.linear2(self.dropout(self.linear1(x)))
    x = self.layer_norm2(x + x)  # Add & Norm

    return x
  
from einops import repeat

class ViT(nn.Module):
  def __init__(self, ch=3, img_size=32, patch_size=8, embed_dim=128,
               n_layers=4, out_dim=10, dropout=0.1, heads=2):
    super(ViT, self).__init__()

    # Attributes
    self.channels = ch
    self.height = img_size
    self.width = img_size
    self.patch_size = patch_size
    self.n_layers = n_layers

    # Patching
    self.patch_embed = PatchEmbedding(patch_size=patch_size,
                                      in_channels=ch,
                                      embed_dim=embed_dim)

    # Learnable Params
    num_patches=(img_size // patch_size) ** 2
    self.pos_embedding = nn.Parameter(
        torch.randn(1, num_patches + 1, embed_dim)
    )
    self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))

    # Transformer Encoder
    self.layers = nn.ModuleList([])
    for _ in range(n_layers):
      transformer = EncoderLayer(d_model=embed_dim, nhead=heads, dropout=dropout)
      self.layers.append(transformer)

    # Classification head
    self.class_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, out_dim))

  def forward(self, img):
    # get patches
    x = self.patch_embed(img)
    b, n, _ = x.shape

    # get cls token
    cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
    x = torch.cat((cls_token, x), dim=1)
    x += self.pos_embedding[:, :(n + 1)]

    # Transformerlayer
    for i in range(self.n_layers):
      x = self.layers[i](x)

    out = self.class_head(x[:, 0, :])
    return out