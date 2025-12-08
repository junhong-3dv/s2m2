# src/core/utils/model_utils.py
import torch
import os
import numpy as np
import open3d as o3d
from ..model.s2m2 import S2M2 as Model
from .image_utils import image_pad, image_crop



def load_model(pretrain_path, model_type, use_positivity=True, refine_iter=3, device=None):
    model_config = {
        "S": {"feature_channels": 128, "n_transformer": 1 * 1},
        "M": {"feature_channels": 192, "n_transformer": 1 * 2},
        "L": {"feature_channels": 256, "n_transformer": 1 * 3},
        "XL": {"feature_channels": 384, "n_transformer": 1 * 3}
    }

    if model_type not in model_config:
        print('model type should be one of [S, M, L, XL]')
        exit(1)

    config = model_config[model_type]
    feature_channels = config["feature_channels"]
    n_transformer = config["n_transformer"]

    model_path = f"CH{feature_channels}NTR{n_transformer}.pth"
    ckpt_path = os.path.join(pretrain_path, model_path)

    model = Model(
        feature_channels=feature_channels,
        dim_expansion=1,
        num_transformer=n_transformer,
        use_positivity=use_positivity,
        refine_iter=refine_iter,
    )

    try:
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.my_load_state_dict(checkpoint["state_dict"])
        model.eval()
        if device:
            model = model.to(device)
        print("Model loaded")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@torch.no_grad()
def run_stereo_matching(model, left_torch, right_torch, device, N_repeat=1):
    """
    Run stereo matching

    Args:
        model: stereo model
        left_torch: left image in torch
        right_torch: right_image
        device: 'cuda:0' if torch.cuda.is_available() else 'cpu'
        N_repeat: number of inference for run-time estimation

    Returns:
        pred_disp: stereo disparity map
        pred_occ: stereo occlusion map (0 means occluded)
        pred_conf: stereo confidence map (1 if disp error < 4px else 0)
        avg_conf_score: avg confidence score
        run_time: avg run time

    """
    # preprocessing
    img_height, img_width = left_torch.shape[-2:]
    left_torch_pad = image_pad(left_torch, 32).to(device)
    right_torch_pad = image_pad(right_torch, 32).to(device)

    with torch.inference_mode():
        with torch.amp.autocast(enabled=True, device_type=device.type, dtype=torch.float16):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(N_repeat):
                pred_disp, pred_occ, pred_conf = model(left_torch_pad, right_torch_pad)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()

    run_time = starter.elapsed_time(ender)
    run_time = run_time / N_repeat

    # post-processing
    pred_disp = image_crop(pred_disp, (img_height, img_width)).squeeze().float()
    pred_occ = image_crop(pred_occ, (img_height, img_width)).squeeze().float()
    pred_conf = image_crop(pred_conf, (img_height, img_width)).squeeze().float()

    margin = 100
    avg_conf_score = pred_conf[margin:-margin, margin:-margin].mean().item()
    return pred_disp, pred_occ, pred_conf, avg_conf_score, run_time


def compute_confidence_score(model, left_torch, right_torch, device):
    """Wrapper func to compute confidence score of stereo matching  """
    pred_disp, pred_occ, pred_conf, avg_conf_score, _ = run_stereo_matching(model, left_torch, right_torch, device, N_repeat=1)
    # opencv 2D visualization
    # pred_disp, pred_occ, pred_conf = pred_disp.cpu().numpy(), pred_occ.cpu().numpy(), pred_conf.cpu().numpy()
    # from .vis_utils import visualize_stereo_results_2d
    # left = left_torch[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # right = right_torch[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # visualize_stereo_results_2d(left, right, pred_disp, pred_occ, pred_conf)
    return avg_conf_score



def get_pointcloud(rgb, disp, calib, depth_trunc=None):

    if depth_trunc is None:
        depth_trunc = 1e9

    h, w = rgb.shape[:2]
    intrinsic = calib['cam0']
    fx = intrinsic[0, 0]/2.0
    cx = intrinsic[0, 2]/2.0
    cy = intrinsic[1, 2]/2.0
    baseline = calib['baseline']
    doffs = calib['doffs']
    print(f"doffs:{doffs}")
    depth = baseline * fx / (disp + doffs)
    depth[disp<=0]=1e9
    depth = o3d.geometry.Image(depth.astype(np.float32))
    rgb = o3d.geometry.Image(rgb)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
                                                              depth,
                                                              depth_scale=1000.0,
                                                              depth_trunc=depth_trunc,
                                                              convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fx, cx, cy)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    return point_cloud
