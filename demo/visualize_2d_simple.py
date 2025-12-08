import argparse
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import torch
import torch._dynamo
from s2m2.core.utils.model_utils import load_model, run_stereo_matching
from s2m2.core.utils.image_utils import read_images
from s2m2.core.utils.vis_utils import visualize_stereo_results_2d

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch._dynamo.config.verbose = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='S', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=3, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--torch_compile', action='store_true', help='apply torch_compile')
    parser.add_argument('--allow_negative', action='store_true', help='allow negative disparity for imperfect rectification')
    return parser


def main(args):
    # load stereo model
    model = load_model(os.path.join(project_root, "weights/pretrain_weights"), args.model_type, not args.allow_negative, args.num_refine, device)
    if args.torch_compile:
        model = torch.compile(model)

    # load web stereo images
    if args.allow_negative:
        left_path = 'Web/64648_pbz98_3D_MPO_70pc_L.jpg'
        right_path = 'Web/64648_pbz98_3D_MPO_70pc_R.jpg'
    else:
        left_path = 'Web/0025_L.png'
        right_path = 'Web/0025_R.png'

    left_path = os.path.join(project_root, "data", "samples", left_path)
    right_path = os.path.join(project_root, "data", "samples", right_path)

    # load stereo images
    left, right = read_images(left_path, right_path)

    img_height, img_width = left.shape[:2]
    print(f"original image size: img_height({img_height}), img_width({img_width})")

    img_height = (img_height // 32) * 32
    img_width = (img_width // 32) * 32
    print(f"cropped image size: img_height({img_height}), img_width({img_width})")

    # image crop
    left = left[:img_height, :img_width]
    right = right[:img_height, :img_width]

    # to torch tensor
    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).to(device)
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).to(device)

    # run stereo matching
    _ = run_stereo_matching(model, left_torch, right_torch, device) #pre-run
    pred_disp, pred_occ, pred_conf, avg_conf_score, avg_run_time = run_stereo_matching(model, left_torch, right_torch, device, N_repeat=5)
    print(F"torch avg inference time:{(avg_run_time)/1000}, FPS:{1000/(avg_run_time)}")

    # opencv 2D visualization
    pred_disp, pred_occ, pred_conf = pred_disp.cpu().numpy(), pred_occ.cpu().numpy(), pred_conf.cpu().numpy()
    visualize_stereo_results_2d(left, right, pred_disp, pred_occ, pred_conf)



if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)
