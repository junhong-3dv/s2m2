import argparse
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import torch
import torch._dynamo
from s2m2.core.utils.model_utils import load_model
from s2m2.core.utils.image_utils import read_images
from s2m2.core.utils.vis_utils import visualize_stereo_results_2d
from s2m2.tools.export_model import export_onnx
import onnxruntime

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
    parser.add_argument('--allow_negative', action='store_true', help='allow negative disparity for imperfect rectification')
    parser.add_argument('--img_height', default=1088, type=int,
                        help='image height')
    parser.add_argument('--img_width', default=800, type=int,
                        help='image width')

    return parser

def main(args):
    img_height, img_width = args.img_height, args.img_width

    model = load_model(os.path.join(project_root, "weights/pretrain_weights"), args.model_type, not args.allow_negative, args.num_refine, 'cpu')

    # load web stereo images
    if args.allow_negative:
        left_path = 'samples/Web/64648_pbz98_3D_MPO_70pc_L.jpg'
        right_path = 'samples/Web/64648_pbz98_3D_MPO_70pc_R.jpg'
    else:
        left_path = 'samples/Web/0025_L.png'
        right_path = 'samples/Web/0025_R.png'

    left_path = os.path.join(project_root, "data", left_path)
    right_path = os.path.join(project_root, "data", right_path)

    # load stereo images
    left, right = read_images(left_path, right_path)

    print(f"image size: {img_height}, {img_width}")

    if left.shape[1]>=img_width and left.shape[0]>=img_height:
        left = left[:img_height, :img_width]
        right = right[:img_height, :img_width]
    else:
        left = cv2.resize(left, dsize=(img_width, img_height))
        right = cv2.resize(right, dsize=(img_width, img_height))


    # to torch tensor
    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0))
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0))

    torch_version = torch.__version__
    onnx_path = os.path.join(project_root, 'weights/onnx_save', f'S2M2_{args.model_type}_{img_width}_{img_height}_v2_torch{torch_version[0]}{torch_version[2]}.onnx')
    print(onnx_path)

    export_onnx(model, onnx_path, left_torch, right_torch)

    # test onnx file with onnxruntime in gpu
    sess = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    print(f"onnxruntime device: {onnxruntime.get_device()}")

    input_name = [input.name for input in sess.get_inputs()]
    output_name = [output.name for output in sess.get_outputs()]
    print(f"onnx_input_name:{input_name}")
    print(f"onnx_output_name:{output_name}")
    outputs = sess.run([output_name[0], output_name[1], output_name[2]],
                          {input_name[0]: left_torch.numpy(),
                           input_name[1]: right_torch.numpy()})
    print(f"onnx_output shape: {outputs[1].shape}")

    pred_disp, pred_occ, pred_conf = outputs
    pred_disp = np.squeeze(pred_disp)
    pred_occ = np.squeeze(pred_occ)
    pred_conf = np.squeeze(pred_conf)

    # opencv 2D visualization
    visualize_stereo_results_2d(left, right, pred_disp, pred_occ, pred_conf)


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)