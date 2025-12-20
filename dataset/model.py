import torch
from torch import nn

from .const import EMBEDDING_DIM


class DinoV2WithNormalize(torch.nn.Module):
    """
    DINOv2 outputs unnormalized embeddings by default

    We normalize them to L2, this is standard choice
    for embeddings because:
      - Embeddings live in continuous semantic space
      - Better geometric properties for nearest neighbor search
      - HNSW performs well with L2 distance
    """

    def __init__(self):
        super().__init__()
        model_variant = {
            384: "dinov2_vits14_reg",
            768: "dinov2_vitb14_reg",
            1024: "dinov2_vitl14_reg",
            1536: "dinov2_vitg14_reg",
        }[EMBEDDING_DIM]
        self.model = torch.hub.load("facebookresearch/dinov2", model_variant)

    def forward(self, samples):
        return nn.functional.normalize(self.model(samples), dim=-1, p=2)
