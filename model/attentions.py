import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor


class SelfAttn(nn.Module):
    """
    Self Attention Module
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            dim_expansion: int,
            use_pe: bool,
    ):
        super(SelfAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pe = use_pe
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)

        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)
        if self.use_pe:
            self.pe_proj = nn.Linear(32, self.head_dim)


    def forward(self, x: torch.Tensor, pe: torch.Tensor=None) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        scale = self.scale

        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # B num_head N head_dim
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_pe:
            score = torch.einsum('...ic, ...jc -> ...ij', scale * q, k)  # B num_head N N
            attn = score.reshape(B, self.num_heads, N, N).softmax(dim=-1)
            out = torch.einsum('...ij, ...jc -> ...ic', attn, v)
            # add contextual relative positional encoding
            pe_sum = torch.einsum('...nij, ijc -> ...nic', attn, pe)
            out = out + self.pe_proj(pe_sum)
        else:
            out = cp.checkpoint(F.scaled_dot_product_attention,q,k,v, use_reentrant=False)

        out = self.proj(out.transpose(1, 2).reshape(B, N, self.num_heads*self.head_dim))

        return out


class CrossAttn(nn.Module):
    """
    Symmetric Cross Attention Module
    """
    def __init__(
            self,
            dim: int,
            num_heads: int,
            dim_expansion: int,
    ):
        super(CrossAttn, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim_expansion * dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.k = nn.Linear(dim, dim_expansion * dim, bias=False)
        self.v = nn.Linear(dim, dim_expansion * dim, bias=True)
        self.proj = nn.Linear(dim_expansion * dim, dim, bias=False)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # x, y: [B, N, C], [B, N, C]
        B, N, C = x.shape
        B, _, C = y.shape

        qx = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        ky = self.k(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vy = self.v(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x_out = F.scaled_dot_product_attention(qx,ky,vy)

        kx = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        qy = self.q(y).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        vx = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        y_out = F.scaled_dot_product_attention(qy,kx,vx)

        x_out = self.proj(x_out.transpose(1, 2).reshape(B, N, self.num_heads*self.head_dim))
        y_out = self.proj(y_out.transpose(1, 2).reshape(B, N, self.num_heads*self.head_dim))

        return x_out, y_out


class SelfAttnBlock1D(nn.Module):
    """
    1D Self Attention Block (pre-norm type)
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dim_expansion: int,
                 use_pe: bool,
                 ):
        super(SelfAttnBlock1D, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.attn = SelfAttn(dim=self.dim,
                             num_heads=self.num_heads,
                             dim_expansion=dim_expansion,
                             use_pe=use_pe)
        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor, pe: torch.Tensor=None):

        B, H, W, C = z.shape
        z = z.reshape(B*H,W,C)

        z_norm = (self.norm_pre(z))
        z = self.attn(z_norm, pe) + z

        z = z.reshape(B,H,W,C)
        return z


class CrossAttnBlock1D(nn.Module):
    """
    1D Cross Attention Block (pre-norm type)
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dim_expansion: int,
                 ):
        super(CrossAttnBlock1D, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.attn = CrossAttn(self.dim,
                              self.num_heads,
                              dim_expansion=dim_expansion)

        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor):

        z_norm = (self.norm_pre(z))
        x, y = z_norm.chunk(2, dim=0)

        B, H, W, C = x.shape
        x, y = x.reshape(B * H, W, C), y.reshape(B * H, W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        z = torch.cat([x, y], dim=0) + z

        return z



class SelfAttnBlock2D(nn.Module):
    """
    2D Self Attention Block (pre-norm type)
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dim_expansion: int,
                 use_pe: bool
                 ):
        super(SelfAttnBlock2D, self).__init__()

        self.dim = dim
        self.attn = SelfAttn(dim=dim,
                             num_heads=num_heads,
                             dim_expansion=dim_expansion,
                             use_pe=use_pe)
        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)


    def forward(self, z: torch.Tensor, pe: torch.Tensor=None):

        B, H, W, C = z.shape
        z = z.reshape(B,H*W,C).contiguous()
        z_norm = (self.norm_pre(z))
        z = self.attn(z_norm, pe) + z
        z = z.reshape(B,H,W,C).contiguous()

        return z



class CrossAttnBlock2D(nn.Module):
    """
    2D Cross Attention Block (pre-norm type)
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dim_expansion: int,
                 ):
        super(CrossAttnBlock2D, self).__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.attn = CrossAttn(self.dim,
                              self.num_heads,
                              dim_expansion=dim_expansion)
        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor):

        z_norm = (self.norm_pre(z))
        x, y = z_norm.chunk(2, dim=0)

        B, H, W, C = x.shape
        x, y = x.reshape(B, H*W, C), y.reshape(B, H*W, C)
        x, y = self.attn(x, y)
        x, y = x.reshape(B, H, W, C), y.reshape(B, H, W, C)
        z = torch.cat([x, y], dim=0) + z

        return z


class FFN(nn.Module):
    """
    Feed Forward Network Block (pre-norm type)
    """
    def __init__(self,
                 dim: int,
                 dim_expansion: int
                 ):
        super(FFN, self).__init__()
        self.dim = dim
        self.ffn = nn.Sequential(nn.Linear(self.dim, dim_expansion * self.dim),
                                 nn.GELU(),
                                 nn.Linear(dim_expansion * self.dim, self.dim))

        self.norm_pre = nn.LayerNorm(self.dim, elementwise_affine=False)

    def forward(self, z: torch.Tensor):
        # z: [B, H, W, C]
        z_norm = self.norm_pre(z)
        z = self.ffn(z_norm) + z

        return z




class ConvBlock2D(nn.Module):
    """
    Conv Block
    """
    def __init__(self,
                 dim: int,
                 kernel_size: int,
                 dim_expansion: int,
                 ):
        super(ConvBlock2D, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.convs = nn.Sequential(nn.Conv2d(self.dim, dim_expansion*self.dim, self.kernel_size, padding=self.kernel_size // 2),
                                    nn.GELU(),
                                    nn.Conv2d(dim_expansion * self.dim, self.dim, self.kernel_size, padding=self.kernel_size // 2))

        self.convs_1x = nn.Sequential(nn.Conv2d(self.dim, dim_expansion*self.dim, 1),
                                    nn.ReLU(),
                                    nn.Conv2d(dim_expansion * self.dim, self.dim, 1))

    def forward(self, z: torch.Tensor):
        # x: [B, C, H, W]
        out = self.convs(z) + self.convs_1x(z)

        return out


class GlobalAttnBlock(nn.Module):
    """
    Global 2D Attentions Block
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dim_expansion: int,
                 use_cross_attn: bool=False,
                 use_pe: bool=False
                 ):
        super(GlobalAttnBlock, self).__init__()

        self.self_attn = SelfAttnBlock2D(dim=dim,
                                         num_heads=num_heads,
                                         dim_expansion=dim_expansion,
                                         use_pe=use_pe)
        if use_cross_attn:
            self.cross_attn = CrossAttnBlock2D(dim=dim,
                                             num_heads=num_heads,
                                             dim_expansion=dim_expansion)
            self.ffn_c = FFN(dim=dim, dim_expansion=dim_expansion)
        else:
            self.cross_attn = None
        self.ffn = FFN(dim=dim, dim_expansion=dim_expansion)


    def forward(self, z: torch.Tensor, pe: torch.Tensor=None):
        # z: [B, HW, C]
        z = z.permute(0,2,3,1)
        if self.cross_attn is not None:
            z = self.cross_attn(z)
            z = self.ffn_c(z)
        z = self.self_attn(z, pe)
        z = self.ffn(z)
        z = z.permute(0,3,1,2)

        return z.contiguous()


class BasicAttnBlock(nn.Module):
    """
    1D Attentions Block
    """
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dim_expansion: int,
                 use_pe: bool=False
                 ):
        super(BasicAttnBlock, self).__init__()

        self.cross_attn = CrossAttnBlock1D(dim=dim,
                                              num_heads=num_heads,
                                              dim_expansion=dim_expansion)

        self.self_attn = SelfAttnBlock1D(dim=dim,
                                            num_heads=num_heads,
                                            dim_expansion=dim_expansion,
                                            use_pe=use_pe)
        self.ffn_c = FFN(dim=dim, dim_expansion=dim_expansion)
        self.ffn = FFN(dim=dim, dim_expansion=dim_expansion)

    def forward(self, z: torch.Tensor, pe: torch.Tensor=None):
        # z: [B, C, H, W]
        z = z.permute(0, 2, 3, 1)
        z = self.cross_attn(z)
        z = self.ffn_c(z)
        z = self.self_attn(z, pe)
        z = self.ffn(z)
        z = z.permute(0, 3, 1, 2)
        return z
