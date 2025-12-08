import argparse
import os
import sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import torch
import torch._dynamo
from s2m2.core.utils.model_utils import load_model
from s2m2.core.utils.image_utils import read_images, rectify_images
from s2m2.core.utils.calib_utils import load_calibration_data, compute_stereo_rectification, create_delta_rotation

from s2m2.calibration.base import vivisualize_calibration_results
from s2m2.calibration.cem import cem_calibration
from s2m2.calibration.grad_descent import gradient_descent_calibration
from s2m2.calibration.keypoint_matching import keypoint_based_calibration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch._dynamo.config.verbose = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib_method', default='CEM', type=str, help='select model type: CEM,GD,KP')
    parser.add_argument('--model_type', default='L', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=3, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--torch_compile', action='store_true', help='torch_compile')
    return parser



def main(args):
    print("This is a demo for online stereo calibration\n")
    print("Under assumption where only extrinsic rotation matrix is slightly twisted\n")

    model = load_model(os.path.join(project_root, "weights/pretrain_weights"), args.model_type, True, args.num_refine, device)
    if args.torch_compile:
        model = torch.compile(model)

    left_path = os.path.join(project_root, "data", "calib/1_10_sensor_raw_left.png")
    right_path = os.path.join(project_root, "data", "calib/1_10_sensor_raw_right.png")
    calib_path = os.path.join(project_root, "data", "calib/1_01_camera_param_head.xml")

    # load stereo data
    left, right = read_images(left_path, right_path)
    calib_data = load_calibration_data(calib_path)

    calib_method = args.calib_method
    if calib_method == 'CEM' or calib_method == 'cem':
        # Perform CEM based online-calibration
        results = cem_calibration(model, left, right, calib_data, device)

    elif calib_method == 'GD' or calib_method == 'gd':
        # Perform gradient-descent based online-calibration
        results = gradient_descent_calibration(model, left, right, calib_data, device)

    elif calib_method == 'KP' or calib_method == 'kp':
        # Perform keypoint-matching based online-calibration
        results = keypoint_based_calibration(left, right, calib_data)
    else:
        print(" method should be in [CEM, GD, KP]")

    if results:
        print("\n Online Calibration Completed")
    else:
        print("\n Online Calibration Failed!")
        return

    # Apply original rectification for comparison
    h, w = left.shape[:2]
    image_size = (w, h)
    try:
        rectification_data = compute_stereo_rectification(calib_data, image_size, None)
        left_rectified_orig, right_rectified_orig = rectify_images(left, right, rectification_data)
        print("Original stereo rectification completed for visualization!")
    except Exception as e:
        print(f"Error during original rectification for visualization: {e}")
        return
    
    # Apply calibration for comparison
    try:
        delta_R = create_delta_rotation(results['roll_delta'], results['pitch_delta'], results['yaw_delta'])
        rectification_data = compute_stereo_rectification(calib_data, image_size, delta_R)
        left_rectified_cem, right_rectified_cem = rectify_images(left, right, rectification_data)
        print("Stereo rectification completed for visualization!")
    except Exception as e:
        print(f"Error during rectification for visualization: {e}")
        return
    
    # Visualize results
    print("\n" + "="*50)
    print("Visualizing online Calibration Results")
    print("="*50)
    vivisualize_calibration_results(
        left_rectified_orig, right_rectified_orig, 
        left_rectified_cem, right_rectified_cem, 
        model, device
    )

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)