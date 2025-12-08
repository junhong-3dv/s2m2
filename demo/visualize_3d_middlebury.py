import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import open3d as o3d
import torch
import torch._dynamo
from s2m2.core.utils.model_utils import load_model, run_stereo_matching
from s2m2.core.utils.image_utils import read_images
from s2m2.core.utils.vis_utils import visualize_stereo_results_2d, visualize_stereo_results_3d

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch._dynamo.config.verbose = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='XL', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=5, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--torch_compile', action='store_true', help='torch_compile')
    return parser

def get_pointcloud(rgb, disp, calib):
    h, w = rgb.shape[:2]
    intrinsic = calib['cam0']
    fx = intrinsic[0, 0]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
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
                                                              depth_trunc=3.0,
                                                              convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fx, cx, cy)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return point_cloud

def read_calib_file(calib_path):
    d = {}
    with open(calib_path, "r") as of:
        lines = of.readlines()
    for line in lines:
        k, v = line.split("=")
        if k == 'cam0' or k == 'cam1':
            v = v[1:-2]
            v = [float(it) for it in v.replace(';', '').split(' ')]
            v = np.array(v).reshape((3, 3))
        else:
            v = float(v[:-1])
        d[k]=v

    calib_dict = d
    return calib_dict


def main(args):

    model = load_model(os.path.join(project_root, "weights/pretrain_weights"), args.model_type, True, args.num_refine, device)
    if args.torch_compile:
        model = torch.compile(model)

    left_path = os.path.join(project_root, "data", "samples/Bicycle2/im0.png")
    right_path = os.path.join(project_root, "data", "samples/Bicycle2/im1.png")
    calib_path = os.path.join(project_root, "data", "samples/Bicycle2/calib.txt")

    # load stereo images
    left, right = read_images(left_path, right_path)

    # load calibration params
    calib = read_calib_file(calib_path)

    # to torch tensor
    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).to(device)
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).to(device)

    # run stereo matching
    _ = run_stereo_matching(model, left_torch, right_torch, device) #pre-run
    pred_disp, pred_occ, pred_conf, avg_conf_score, avg_run_time = run_stereo_matching(model, left_torch, right_torch, device, N_repeat=5)
    print(F"torch avg inference time:{(avg_run_time)/1000}, FPS:{1000/(avg_run_time)}")

    # opencv 2D visualization
    pred_disp, pred_occ, pred_conf = pred_disp.cpu().numpy(), pred_occ.cpu().numpy(), pred_conf.cpu().numpy()
    disp_left_vis, disp_left_vis_mask, valid = visualize_stereo_results_2d(left, right, pred_disp, pred_occ, pred_conf)

    # open3d pointcloud visualization
    pred_disp_np = np.ascontiguousarray(pred_disp).astype(np.float32)
    pred_disp_np_filt = pred_disp_np * valid
    pred_disp_np_filt[~valid] = -1

    pcd = get_pointcloud(left, pred_disp_np, calib)
    pcd_filt = get_pointcloud(left, pred_disp_np_filt, calib)
    visualize_stereo_results_3d(pcd, pcd_filt)


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)



