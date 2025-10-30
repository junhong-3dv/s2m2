import torch.nn as nn
import torch
from torch import Tensor
class FeatureFusion(nn.Module):
    def __init__(self,
                 dim: int,
                 kernel_size: int,
                 use_gate=True,
                 ):
        super(FeatureFusion, self).__init__()

        pad = kernel_size//2
        self.use_gate = use_gate
        if use_gate:
            self.feature_gate = nn.Sequential(nn.Conv2d(2*dim, dim, kernel_size=kernel_size, padding=pad),
                                              nn.GELU(),
                                              nn.Conv2d(dim, dim, kernel_size=1),
                                              nn.Sigmoid())
        self.feature_fusion = nn.Sequential(nn.Conv2d(2*dim, 2*dim, kernel_size=kernel_size, padding=pad),
                                          nn.GELU(),
                                          nn.Conv2d(2*dim, dim, kernel_size=1))


    def forward(self, z0: Tensor, z1: Tensor):
        z = torch.cat([z0,z1],dim=1)
        if self.use_gate:
            eps=0.01
            w = self.feature_gate(z).clamp(min=eps, max=1-eps)
            z_out = self.feature_fusion(z) + (w) * z0 + (1 - w) * z1
        else:
            z_out = self.feature_fusion(z)

        return z_out
