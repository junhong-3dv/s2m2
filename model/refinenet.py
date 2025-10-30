import torch.nn as nn
import torch
from torch import Tensor
from typing import Callable
from .unet import Unet
import torch.nn.functional as F
class ConvGRU(nn.Module):
    def __init__(self,
                 hidden_dim: int=128,
                 input_dim: int=128,
                 kernel_size: int=3
                 ):
        super(ConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [kernel_size,1], padding=[kernel_size // 2, 0])
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [kernel_size,1], padding=[kernel_size // 2, 0])
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [kernel_size,1], padding=[kernel_size // 2, 0])

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [1,kernel_size], padding=[0, kernel_size // 2])
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [1,kernel_size], padding=[0, kernel_size // 2])
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, [1,kernel_size], padding=[0, kernel_size // 2])

    def forward(self, h: Tensor, x: Tensor):

        hx = torch.cat([h,x],dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = F.tanh(self.convq1(torch.cat([r * h, x], axis=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = F.tanh(self.convq2(torch.cat ([r * h, x], axis=1)))
        h = (1 - z) * h + z * q

        return h.to(x.dtype)


class GlobalRefiner(nn.Module):
    """
    refine disparity in global manner without local feature correlations
    """
    def __init__(self,
                 feature_channels: int):
        super(GlobalRefiner, self).__init__()

        self.init_feat = nn.Sequential(nn.Conv2d(2 + feature_channels, feature_channels, kernel_size=3, padding=1),
                                       nn.GELU(),
                                       nn.Conv2d(feature_channels, feature_channels, kernel_size=1))

        self.refine_unet = Unet(dims=[feature_channels, 1 * feature_channels, 1 * feature_channels],
                                dim_expansion=1,
                                use_pe=False,
                                use_gate_fusion=True,
                                n_attn=1)

        self.out_feat = nn.Sequential(nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1))

    def forward(self,
                ctx: Tensor,
                disp: Tensor,
                conf: Tensor) -> Tensor:
        # image_width = disp.shape[-1]
        disp_nor = disp/1e2
        mask = 1.0 * (conf > .2)
        conf_logit = (mask*conf).logit(eps=1e-1)

        feat = self.init_feat(torch.cat([disp_nor*mask, conf_logit, ctx],dim=1).to(disp.dtype))
        refine_feat = self.refine_unet(feat)[0]
        # out = self.out_feat(refine_feat)
        # disp_update = out*image_width
        disp_update = self.out_feat(refine_feat)*1e2
        disp_out = (mask*disp + (1-mask)*disp_update).to(disp.dtype)

        return disp_out


class LocalRefiner(nn.Module):
    """
    refine disparity in local iterative manner using local feature correlations
    """
    def __init__(self,
                 feature_channels: int,
                 dim_expansion: int,
                 radius: int,
                 use_gate_fusion: bool):
        super(LocalRefiner, self).__init__()

        self.disp_feat = nn.Sequential(nn.Conv2d(1, 96, kernel_size=3, padding=1),
                                       nn.GELU(),
                                       nn.Conv2d(96, 96, kernel_size=3, padding=1))

        self.corr_feat1 = nn.Sequential(nn.Conv2d((2 * radius + 1) * 1, 96, kernel_size=1),
                                        nn.GELU(),
                                        nn.Conv2d(96, 64, kernel_size=1))
        self.corr_feat2 = nn.Sequential(nn.Conv2d((2 * radius + 1) * 1, 96, kernel_size=1),
                                        nn.GELU(),
                                        nn.Conv2d(96, 64, kernel_size=1))


        self.conf_occ_feat = nn.Sequential(nn.Conv2d(2, 64, kernel_size=3,padding=1),
                                       nn.GELU(),
                                       nn.Conv2d(64, 32, kernel_size=1))

        self.disp_corr_ctx_cat = nn.Sequential(nn.Conv2d(256 + feature_channels, 2*feature_channels, kernel_size=1),
                                      nn.GELU(),
                                      nn.Conv2d(2*feature_channels, feature_channels, kernel_size=3, padding=1))


        self.refine_unet = Unet(dims=[feature_channels, 1 * feature_channels, 2 * feature_channels],
                                dim_expansion=dim_expansion,
                                use_pe=False,
                                use_gate_fusion=use_gate_fusion,
                                n_attn=1)


        self.disp_update = nn.Sequential(nn.Conv2d(feature_channels, 1*feature_channels, kernel_size=3, padding=1),
                                         nn.GELU(),
                                         nn.Conv2d(1*feature_channels, 1, kernel_size=3, padding=1, bias=False))


        self.conf_occ_update = nn.Sequential(nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
                                         nn.GELU(),
                                         nn.Conv2d(feature_channels, 2, kernel_size=3, padding=1, bias=False))

        self.gru = ConvGRU(feature_channels, feature_channels, 3)

    def forward(self,
                hidden: Tensor,
                ctx: Tensor,
                disp: Tensor,
                conf: Tensor,
                occ: Tensor,
                cv_fn: Callable) -> Tensor:

        conf_logit = conf.logit(eps=1e-2)
        occ_logit = occ.logit(eps=1e-2)

        corr1, corr2 = cv_fn(disp)
        corr_feat1 = self.corr_feat1(corr1 / 16)
        corr_feat2 = self.corr_feat2(corr2 / 16)
        disp_feat = self.disp_feat(disp/1e2)
        conf_feat = self.conf_occ_feat(torch.cat([conf_logit, occ_logit],dim=1).to(disp.dtype))
        disp_corr_ctx_feat = self.disp_corr_ctx_cat(torch.cat([disp_feat, corr_feat1, corr_feat2, ctx, conf_feat], dim=1).to(disp.dtype))

        refine_feat = self.refine_unet(disp_corr_ctx_feat)[0]
        hidden_new = self.gru(hidden, refine_feat)
        disp_update = self.disp_update(hidden_new)
        conf_update, occ_update = self.conf_occ_update(hidden_new).chunk(2, dim=1)

        conf_new = torch.sigmoid(conf_update + conf_logit).to(disp.dtype)
        occ_new = torch.sigmoid(occ_update + occ_logit).to(disp.dtype)
        disp_new = (disp + disp_update).to(disp.dtype)


        return hidden_new.to(disp.dtype), disp_new.to(disp.dtype), conf_new.to(disp.dtype), occ_new.to(disp.dtype)