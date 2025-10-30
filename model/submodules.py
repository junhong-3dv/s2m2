import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import bilinear_sampler


class CostVolume():
    """
    cost volume for iterative refinemtns
    """
    def __init__(self, cv: Tensor, cv_2x: Tensor, coords: Tensor, radius: int):
        self.radius = radius

        r = self.radius
        dx = torch.linspace(-r, r, 2 * r + 1, device=cv.device, dtype=cv.dtype)
        self.dx = dx.reshape(1, 1, 2 * r + 1, 1)

        b, h, w, w2 = cv.shape
        self.cv = cv.reshape(b * h * w, 1, 1, w2)
        self.cv_2x = F.interpolate(cv_2x.permute(0,3,1,2), scale_factor=2)\
            .permute(0,2,3,1).reshape(b * h * w, 1, 1, w2//2)

        self.coords = coords.reshape(b * h * w, 1, 1, 1)

    def __call__(self, disp: Tensor):
        b, _, h, w = disp.shape
        dx = self.dx
        x0 = self.coords - disp.reshape(b * h * w, 1, 1, 1) + dx
        y0 = 0 * x0
        init_coords_lvl = torch.cat([x0, y0], dim=-1)
        corrs = bilinear_sampler(self.cv, init_coords_lvl)
        corrs = corrs.reshape(b, h, w, 2*self.radius+1).permute(0, 3, 1, 2)

        dx = self.dx
        x0 = self.coords / 2 - disp.reshape(b * h * w, 1, 1, 1) / 2 + dx
        y0 = 0 * x0
        init_coords_lvl = torch.cat([x0, y0], dim=-1)
        corrs_2x = bilinear_sampler(self.cv_2x, init_coords_lvl)
        corrs_2x = corrs_2x.reshape(b, h, w, 2*self.radius+1).permute(0, 3, 1, 2)

        return corrs, corrs_2x


class CNNEncoder(nn.Module):
    """
    init convolution neural networks for feature extraction
    """
    def __init__(self, output_dim: int):
        super(CNNEncoder, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=1),
                                   nn.GELU(),
                                   nn.Conv2d(16, 16, kernel_size=1))
        self.conv1_down = nn.Sequential(nn.Conv2d(16, 64, kernel_size=5, stride=2, padding=2),
                                        nn.GELU(),
                                        nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1))


        self.norm1 = nn.GroupNorm(8, output_dim)

        self.conv2 = nn.Sequential(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1),
                                   nn.GELU(),
                                   nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1))

        self.conv2_down = nn.Sequential(nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1))

    def forward(self, x: torch.Tensor):
        x = (self.conv0(x))
        x_2x = self.norm1(self.conv1_down(x))
        x_2x = self.conv2(x_2x) + x_2x
        x_4x = self.conv2_down(x_2x)
        return x_4x, x_2x


class UpsampleMask4x(nn.Module):
    """
    4x upsampling weights generation
    """
    def __init__(self,
                 dim: int
                 ):
        super(UpsampleMask4x, self).__init__()
        self.conv_x = nn.ConvTranspose2d(dim, 64, kernel_size=2, stride=2)
        self.conv_y = nn.Conv2d(dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv_concat = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=False),
                                         nn.ConvTranspose2d(128, 9, kernel_size=2, stride=2))

    def forward(self, feat_x: torch.Tensor, feat_y: torch.Tensor):
        feat_x = self.conv_x(feat_x)
        feat_y = self.conv_y(feat_y)
        upmask = self.conv_concat(torch.cat([feat_x, feat_y], dim=1))

        return upmask


class UpsampleMask1x(nn.Module):
    """
    edge-guided disparity refinement
    """
    def __init__(self,
                 dim: int,
                 ):
        super(UpsampleMask1x, self).__init__()

        self.conv_disp = nn.Sequential(nn.ConvTranspose2d(1, 16, kernel_size=3, padding=1),
                                       nn.ReLU(inplace=False))
        self.conv_rgb = nn.Sequential(nn.ConvTranspose2d(3, 16, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=False))
        self.conv_ctx = nn.ConvTranspose2d(dim, 16, kernel_size=2, stride=2)

        self.conv_concat = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=False),
                                         nn.ConvTranspose2d(48, 9, kernel_size=1))

    def forward(self, disp: torch.Tensor, rgb: torch.Tensor, ctx: torch.Tensor):
        feat_disp = self.conv_disp(disp)
        feat_rgb = self.conv_rgb(rgb)
        feat_ctx = self.conv_ctx(ctx)

        feat_cat = torch.cat([feat_disp, feat_rgb, feat_ctx], dim=1)
        mask = self.conv_concat(feat_cat)

        return mask


class DispInit(nn.Module):
    """
    Initial Disparity Estimation using Optimal Transport
    """
    def __init__(self,
                 dim: int,
                 ot_iter: int,
                 use_positivity: bool):

        super(DispInit, self).__init__()

        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=True)
        self.ot_iter = ot_iter
        self.use_positivity = use_positivity

    def _sinkhorn(self, attn: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor):
        v = log_nu - torch.logsumexp((attn), dim=2)
        u = log_mu - torch.logsumexp((attn + v.unsqueeze(2)), dim=3)

        for idx in range(self.ot_iter - 1):
            v = log_nu - torch.logsumexp((attn + u.unsqueeze(3)), dim=2)
            u = log_mu - torch.logsumexp((attn + v.unsqueeze(2)), dim=3)

        out = (attn + u.unsqueeze(3) + v.unsqueeze(2))

        return out

    def _optimal_transport(self, attn: torch.Tensor):
        bs, h, w, _ = attn.shape
        dtype = attn.dtype
        # set marginal to be uniform distribution
        marginal = torch.cat([torch.ones([w], device=attn.device), torch.tensor([w], device=attn.device)]) / (2 * w)
        log_mu = marginal.log().reshape(1, 1, w + 1)
        log_nu = marginal.log().reshape(1, 1, w + 1)

        # add dustbins
        p2d = (0, 1, 0, 1)  # pad last dim by (0, 1) and 2nd to last by (0, 1)
        attn = F.pad(attn, p2d, "constant", 0)

        # sinkhorn
        attn = self._sinkhorn(attn, log_mu, log_nu)
        # convert back from log space, recover probabilities by normalization 2W
        w_tensor = torch.tensor(w, dtype=dtype,device=attn.device)
        log_const = torch.log(2*w_tensor)
        attn = (attn[:, :, :-1, :-1] + log_const).exp().to(dtype)
        return attn

    def forward(self,
                feature: Tensor,
                ):

        dtype = feature.dtype
        device = feature.device
        w = feature.shape[-1]
        x_grid = torch.linspace(0, w - 1, w, device=device, dtype=feature.dtype)
        if self.use_positivity:
            mask = torch.triu(torch.ones((w, w), dtype=torch.bool, device=device), diagonal=1)
        else:
            mask = torch.zeros((w, w), dtype=torch.bool, device=device)

        feature0, feature1 = self.layer_norm(feature.permute(0, 2, 3, 1)).chunk(2, dim=0)
        cv = torch.einsum('...hic,...hjc -> ...hij', feature0, feature1)
        feature0_down, feature1_down = F.interpolate(feature0.permute(0, 3, 1, 2), scale_factor=1 / 2), \
                                       F.interpolate(feature1.permute(0, 3, 1, 2), scale_factor=1 / 2)
        cv_down = torch.einsum('...chi,...chj -> ...hij', feature0_down, feature1_down).to(dtype)


        cv_mask = cv.masked_fill(mask, -torch.inf)
        prob = self._optimal_transport(cv_mask)
        masked_prob = prob.masked_fill(mask, 0)

        # estimate hard disparity
        prob_max_ind = (masked_prob).max(dim=-1)[1].unsqueeze(3)
        prob_l = 2
        p1d = (prob_l, prob_l)  # pad last dim by 1 on each side
        masked_prob_pad = F.pad(masked_prob, p1d, "constant", 0)
        conf = 0
        correspondence_left = 0
        for idx in range(2 * prob_l + 1):
            weight = torch.gather(masked_prob_pad, index=prob_max_ind + idx, dim=-1)
            conf += weight
            correspondence_left += weight * (prob_max_ind + idx - prob_l)
        eps = 1e-4
        correspondence_left = (correspondence_left + eps) / (conf + eps)
        disparity = (x_grid.reshape(1, 1, w) - correspondence_left.squeeze(3)).unsqueeze(1)
        conf = conf.unsqueeze(1).squeeze(-1)
        occ = masked_prob.sum(dim=-1).unsqueeze(1)

        return disparity, conf, occ, cv, cv_down


