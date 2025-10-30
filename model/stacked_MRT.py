from torch import Tensor
import torch.nn as nn
from .attentions import BasicAttnBlock, GlobalAttnBlock
from .feature_fusion import FeatureFusion

class MRT(nn.Module):
    """
    Multi Resolution Transformer
    """
    def __init__(self,
                 dims: list,
                 num_heads: int,
                 dim_expansion: int,
                 use_gate_fusion: bool
                 ):

        super(MRT, self).__init__()
        self.num_heads = num_heads
        self.dims = dims


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


        self.down_concat1 = FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.down_concat2 = FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)
        self.down_concat3 = FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.up_concat0 = FeatureFusion(dims[0], kernel_size=1, use_gate=use_gate_fusion)
        self.up_concat1 = FeatureFusion(dims[1], kernel_size=1, use_gate=use_gate_fusion)
        self.up_concat2 = FeatureFusion(dims[2], kernel_size=1, use_gate=use_gate_fusion)

        self.enc_attn0 = BasicAttnBlock(dim=dims[0],
                                        num_heads=1*num_heads,
                                        dim_expansion=dim_expansion,
                                        use_pe=False)
        self.enc_attn1 = BasicAttnBlock(dim=dims[1],
                                        num_heads=2*num_heads,
                                        dim_expansion=dim_expansion,
                                        use_pe=False)
        self.enc_attn2 = BasicAttnBlock(dim=dims[2],
                                        num_heads=4*num_heads,
                                        dim_expansion=dim_expansion,
                                        use_pe=False)
        self.enc_attn3s = nn.ModuleList()
        for i in range(2):
            self.enc_attn3s.append(GlobalAttnBlock(dim=dims[2],
                                         num_heads=8*num_heads,
                                         dim_expansion=dim_expansion,
                                         use_cross_attn=True,
                                         use_pe=False))

        self.dec_attn0 = BasicAttnBlock(dim=dims[0],
                                        num_heads=1*num_heads,
                                        dim_expansion=dim_expansion,
                                        use_pe=False)

        self.dec_attn1 = BasicAttnBlock(dim=dims[1],
                                        num_heads=2*num_heads,
                                        dim_expansion=dim_expansion,
                                        use_pe=False)

        self.dec_attn2 = BasicAttnBlock(dim=dims[2],
                                        num_heads=4*num_heads,
                                        dim_expansion=dim_expansion,
                                        use_pe=False)

        self.dec_attn3s = nn.ModuleList()
        for i in range(2):
            self.dec_attn3s.append(GlobalAttnBlock(dim=dims[2],
                                         num_heads=8*num_heads,
                                         dim_expansion=dim_expansion,
                                         use_cross_attn=True,
                                         use_pe=False))


    def forward(self,
                z0: Tensor,
                z1: Tensor,
                z2: Tensor,
                z3: Tensor,
                ):
        # z0: [B, C, H, W]

        # Encoder
        # enc level 0
        z0 = self.enc_attn0(z0)
        z1 = self.down_concat1(z1, self.down_conv0(z0))
        # enc level 1
        z1 = self.enc_attn1(z1)
        z2 = self.down_concat2(z2, self.down_conv1(z1))

        # enc level 2
        z2 = self.enc_attn2(z2)
        z3 = self.down_concat3(z3, self.down_conv2(z2))

        # enc level 3
        for block in self.enc_attn3s:
            z3 = block(z3)

        # Decoder
        # dec level 3
        for block in self.dec_attn3s:
            z3 = block(z3)

        # dec level 2
        z3_up = self.up_conv2(z3)
        z2 = self.up_concat2(z2, z3_up)
        z2 = self.dec_attn2(z2)

        # dec level 1
        z2_up = self.up_conv1(z2)
        z1 = self.up_concat1(z1, z2_up)
        z1 = self.dec_attn1(z1)

        # dec level 0
        z1_up = self.up_conv0(z1)
        z0 = self.up_concat0(z0, z1_up)
        z0 = self.dec_attn0(z0)

        return z0, z1, z2, z3

class StackedMRT(nn.Module):
    """
    Stacked Multi Resolution Transformer
    """
    def __init__(self,
                 num_transformer: int,
                 dims: list,
                 num_heads: int,
                 dim_expansion: int,
                 use_gate_fusion: bool
                 ):
        super(StackedMRT, self).__init__()

        uformer_list = nn.ModuleList()
        for ii in range(num_transformer):
            uformer_list.append(MRT(dims=dims,
                                        num_heads=num_heads,
                                        dim_expansion=dim_expansion,
                                        use_gate_fusion=use_gate_fusion))
        self.uformer_list = uformer_list

    def forward(self,
                z0: Tensor,
                z1: Tensor,
                z2: Tensor,
                z3: Tensor
                ):
        for l, uformer in enumerate(self.uformer_list):
            z0,z1,z2,z3 = uformer(z0,z1,z2,z3)


        return z0.contiguous()

        
        

