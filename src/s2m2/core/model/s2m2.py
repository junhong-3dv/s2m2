import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .stacked_MRT import StackedMRT
from .submodules import CNNEncoder,  DispInit,  CostVolume, UpsampleMask4x, UpsampleMask1x
from .feature_fusion import FeatureFusion
from .unet import Unet
from .refinenet import LocalRefiner, GlobalRefiner
from .utils import custom_unfold

class S2M2(nn.Module):
    def __init__(self,
                 feature_channels: int,
                 dim_expansion: int,
                 num_transformer: int,
                 use_positivity: bool=False,
                 output_upsample: bool=False,
                 refine_iter: int=3
                 ):
        super(S2M2, self).__init__()

        self.feature_channels = feature_channels
        self.num_transformer = num_transformer
        self.use_positivity = use_positivity
        self.refine_iter = refine_iter
        self.output_upsample = output_upsample

        # CNN feature
        self.cnn_backbone = CNNEncoder(output_dim=feature_channels)

        # Feature Pyramid
        self.feat_pyramid = Unet(dims=[feature_channels, 1 * feature_channels, 2 * feature_channels],
                                 dim_expansion=dim_expansion,
                                 use_gate_fusion=True,
                                 use_pe=True,
                                 n_attn=num_transformer*2)
        # Transformer
        self.transformer = StackedMRT(num_transformer=num_transformer,
                                          dims=[feature_channels, 1*feature_channels, 2*feature_channels],
                                          num_heads=1,
                                          dim_expansion=dim_expansion,
                                          use_gate_fusion=True)

        # Disparity matching using Optimal Transport
        self.disp_init = DispInit(dim=feature_channels, ot_iter=3, use_positivity=use_positivity)

        # upsampling mask
        self.upsample_mask_1x = UpsampleMask1x(feature_channels)
        self.upsample_mask_4x_refine = UpsampleMask4x(feature_channels)



        # global refiner
        self.global_refiner = GlobalRefiner(feature_channels=feature_channels)

        # iterative local refiner
        self.feat_fusion_layer = FeatureFusion(dim=feature_channels, kernel_size=3, use_gate=True)
        self.refiner = LocalRefiner(feature_channels=feature_channels,
                                   dim_expansion=dim_expansion,
                                   radius=4,
                                   use_gate_fusion=True)

        self.ctx_feat = nn.Sequential(nn.Conv2d(1*feature_channels, 1*feature_channels, kernel_size=1),
                                      nn.GELU(),
                                      nn.Conv2d(1*feature_channels, feature_channels, kernel_size=1))

    def my_load_state_dict(self, state_dict):
        model_state_dict = self.state_dict()
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                          f"required shape: {model_state_dict[k].shape}, "
                          f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
        self.load_state_dict(state_dict, strict=False)

    def normalize_img(self, img0, img1):
        '''
        loaded images are in [0, 255]
        img0 (b,3,h,w)
        img1 (b,3,h,w)
        '''
        img0 = (img0 / 255. - .5) * 2
        img1 = (img1 / 255. - .5) * 2

        return img0, img1

    def extract_feature(self, img0: Tensor, img1: Tensor):
        '''
        img0 (b,3,h,w)
        img1 (b,3,h,w)
        '''
        feature, feature_2x = self.cnn_backbone(torch.cat([img0, img1], dim=0))

        return feature, feature_2x


    def upsample4x(self, x: Tensor, up_weights: Tensor):
        '''
        x (b,1,h,w)
        up_weights (b,9,4*h,4*w)
        '''

        b, c, h, w = x.shape
        x_unfold = custom_unfold(x.reshape(b, c, h, w), 3, 1) # onnx compatible
        x_unfold = F.interpolate(x_unfold, (h * 4, w * 4), mode='nearest').reshape(b, 9, h * 4, w * 4)
        up_weights = up_weights.softmax(dim=1)
        eps = 0.1

        up_weights = (up_weights - eps).clamp(min=0)
        norm = up_weights.sum(dim=1, keepdim=True)
        up_weights = (up_weights/norm)

        x_up = (x_unfold * up_weights).sum(1, keepdim=True)

        return x_up

    def upsample1x(self, disp: Tensor, filter_weights: Tensor):
        '''
        x (b,1,h,w)
        filter_weights (b,9,4*h,4*w)
        '''

        disp_unfold = custom_unfold(disp, 3, 1)
        if self.output_upsample:
            upsample_factor=2
            disp_unfold = F.interpolate(disp_unfold, scale_factor=upsample_factor, mode='nearest')
            filter_weights = F.interpolate(filter_weights, scale_factor=upsample_factor, mode='bilinear', align_corners=False)
            filter_weights = filter_weights.softmax(dim=1).to(disp.dtype)

        else:
            filter_weights = filter_weights.softmax(dim=1)
        eps=0.1
        filter_weights = (filter_weights - eps).clamp(min=0)
        norm = filter_weights.sum(dim=1, keepdim=True)
        filter_weights = (filter_weights/norm)

        disp_out = (disp_unfold * filter_weights).sum(1, keepdim=True)
        return disp_out


    def forward(self,
                img0: Tensor,
                img1: Tensor):

        img0_nor, img1_nor = self.normalize_img(img0, img1)

        # CNN feature extraction
        feature_4x, feature_2x = self.extract_feature(img0_nor, img1_nor)  # list of features
        feature0_2x, _ = feature_2x.chunk(2, dim=0)

        # feature pyramid by unet
        feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x = self.feat_pyramid(feature_4x)

        # Multi-res Transformer
        feature_tr_4x = self.transformer(feature_py_4x, feature_py_8x, feature_py_16x, feature_py_32x)

        # Initial disparity/confidence/occlusion estimation
        disp, conf, occ, cv, cv_down = self.disp_init(feature_tr_4x)

        feature0_tr_4x, feature1_tr_4x = feature_tr_4x.chunk(2, dim=0)
        feature0_py_4x, feature1_py_4x = feature_py_4x.chunk(2, dim=0)

        # disparity global refinement
        disp = self.global_refiner(feature0_tr_4x.contiguous(), disp.detach(), conf.detach())
        if self.use_positivity:
            disp = disp.clamp(min=0)

        # iterative local refinement
        feature0_fusion_4x = self.feat_fusion_layer(feature0_tr_4x, feature0_py_4x)
        ctx0 = self.ctx_feat(feature0_fusion_4x)
        hidden = torch.tanh(ctx0)

        b, c, h, w = feature0_fusion_4x.shape
        # w = torch.Tensor([w])
        # b = torch.Tensor([b])
        # h = torch.Tensor([h])
        coords_4x = torch.arange(w, device=feature0_fusion_4x.device, dtype=torch.float32).to(feature0_fusion_4x.dtype)
        cv_fn = CostVolume(cv, cv_down, coords_4x.reshape(1, 1, w, 1).repeat(b, h, 1, 1), radius=4)

        for itr in range(self.refine_iter):
            hidden, disp, conf, occ = self.refiner(hidden, ctx0, disp, conf, occ, cv_fn)
            if self.use_positivity:
                disp = disp.clamp(min=0)
            occ_mask = torch.ge((coords_4x.reshape(1, 1, 1, -1) - disp), 0)
            occ = occ * occ_mask


        # 4x upsampling
        upsample_mask = self.upsample_mask_4x_refine(hidden, feature0_2x)
        disp_up = self.upsample4x(disp * 4, upsample_mask)
        occ_up = self.upsample4x(occ, upsample_mask)
        conf_up = self.upsample4x(conf, upsample_mask)

        # edge guided sharpen
        filter_weights = self.upsample_mask_1x(disp_up, img0_nor, feature0_2x)
        disp_up = self.upsample1x(disp_up, filter_weights)
        occ_up = self.upsample1x(occ_up, filter_weights)
        conf_up = self.upsample1x(conf_up, filter_weights)
        if self.output_upsample:
            disp_up = 2*disp_up

        return disp_up, occ_up, conf_up




