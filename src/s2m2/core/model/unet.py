import torch
import torch.nn as nn
from torch import Tensor
from .attentions import GlobalAttnBlock, ConvBlock2D
from .feature_fusion import FeatureFusion
from .utils import get_pe


class Unet(nn.Module):
    """
    Unet
    """
    def __init__(self,
                 dims: list,
                 dim_expansion: int,
                 use_pe: bool,
                 n_attn: int = 1,
                 use_gate_fusion: bool = True,
                 ):
        super(Unet, self).__init__()
        self.dims=dims
        self.use_pe=use_pe


        self.down_conv0 = nn.Sequential(nn.AvgPool2d(2),
                                      nn.Conv2d(dims[0], dims[1], kernel_size=1))
        self.down_conv1 = nn.Sequential(nn.AvgPool2d(2),
                                      nn.Conv2d(dims[1], dims[2], kernel_size=1))
        self.down_conv2 = nn.Sequential(nn.AvgPool2d(2),
                                      nn.Conv2d(dims[2], dims[2], kernel_size=1))

        self.up_conv0 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                      nn.Conv2d(dims[1], dims[0], kernel_size=1))
        self.up_conv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                      nn.Conv2d(dims[2], dims[1], kernel_size=1))
        self.up_conv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                       nn.Conv2d(dims[2], dims[2], kernel_size=1))

        self.concat_conv0 = FeatureFusion(dims[0], kernel_size=1, use_gate=use_gate_fusion)
        self.concat_conv1 = FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.concat_conv2 = FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.enc0 = ConvBlock2D(dim=dims[0], kernel_size=3, dim_expansion=dim_expansion)
        self.enc1 = ConvBlock2D(dim=dims[1], kernel_size=3, dim_expansion=dim_expansion)
        self.enc2 = ConvBlock2D(dim=dims[2], kernel_size=3, dim_expansion=dim_expansion)
        self.enc3s = nn.ModuleList()
        for i in range(n_attn):
            self.enc3s.append(GlobalAttnBlock(dim=dims[2],
                                   num_heads=8,
                                   dim_expansion=dim_expansion,
                                   use_cross_attn=False,
                                   use_pe=use_pe))

        self.dec0 = ConvBlock2D(dim=dims[0], kernel_size=3, dim_expansion=dim_expansion)
        self.dec1 = ConvBlock2D(dim=dims[1], kernel_size=3, dim_expansion=dim_expansion)
        self.dec2 = ConvBlock2D(dim=dims[2], kernel_size=3, dim_expansion=dim_expansion)
        self.dec3s = nn.ModuleList()
        for i in range(n_attn):
            self.dec3s.append(GlobalAttnBlock(dim=dims[2],
                                   num_heads=8,
                                   dim_expansion=dim_expansion,
                                   use_cross_attn=False,
                                   use_pe=False))

    def forward(self, z: Tensor):
        # z: [B, C, H, W]
        if self.use_pe:
            H,W = z.shape[-2:]
            pe = get_pe(H//8, W//8, 32, z.dtype, z.device)
        else:
            pe=None

        # Encoder
        # enc level 0
        z0 = self.enc0(z)
        z1 = self.down_conv0(z0)

        # enc level 1
        z1 = self.enc1(z1)
        z2 = self.down_conv1(z1)

        # enc level 2
        z2 = self.enc2(z2)
        z3 = self.down_conv2(z2)

        # enc level 3
        for block in self.enc3s:
            z3 = block(z3, pe)

        # Decoder
        # dec level 3
        for block in self.dec3s:
            z3 = block(z3, pe)
        z3_new = z3

        # Decoder
        # dec level 2
        z2_new = self.up_conv2(z3_new)
        z2_new = self.concat_conv2(z2, z2_new)
        z2_new = self.dec2(z2_new)

        # dec level 1
        z1_new = self.up_conv1(z2_new)
        z1_new = self.concat_conv1(z1, z1_new)
        z1_new = self.dec1(z1_new)

        # dec level 0
        z0_new = self.up_conv0(z1_new)
        z0_new = self.concat_conv0(z0, z0_new)
        z0_new = self.dec0(z0_new)

        return z0_new, z1_new, z2_new, z3_new







