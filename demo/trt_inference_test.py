import tensorrt as trt
import argparse
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import torch
import torch._dynamo
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
    parser.add_argument('--model_type', type=str, default='S')
    parser.add_argument('--img_width', type=int, default=800)
    parser.add_argument('--img_height', type=int, default=1088)
    parser.add_argument('--precision', type=str, choices=['fp16', 'tf32', 'fp32'], default='fp16')
    return parser

def main(args):
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    runtime = trt.Runtime(TRT_LOGGER)
    print(trt.__version__)
    trt_path = os.path.join(project_root, f"weights/trt_save/S2M2_{args.model_type}_{args.img_width}_{args.img_height}_{args.precision}.engine")
    print(trt_path)
    with open(trt_path, 'rb') as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()

    img_height, img_width = args.img_height, args.img_width

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # load stereo images
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
    input_left = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).to(device).contiguous()
    input_right = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).to(device).contiguous()

    def torch_gpu_alloc(shape, dtype=torch.float32):
        t = torch.empty(shape, device="cuda", dtype=dtype).contiguous()
        return t

    out1 = torch_gpu_alloc((1, 1, img_height, img_width))
    out2 = torch_gpu_alloc((1, 1, img_height, img_width))
    out3 = torch_gpu_alloc((1, 1, img_height, img_width))

    bindings = [
        input_left.data_ptr(), # input 0
        input_right.data_ptr(), # input 1
        out1.data_ptr(), # output 0
        out2.data_ptr(), # output 1
        out3.data_ptr() # output 2
    ]


    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    T = 100
    context.execute_v2(bindings=bindings)
    context.execute_v2(bindings=bindings)
    starter.record()
    for i in range(T):
        context.execute_v2(bindings=bindings)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    print(F"torch avg inference time:{(curr_time) / T / 1000}, FPS:{1000 * T / (curr_time)}")


    print("Output shapes:", out1.shape, out2.shape, out3.shape)
    print("Output dtype:", out1.dtype)


    pred_disp = np.squeeze(out1.contiguous().cpu().numpy()).astype(np.float32)
    pred_occ = np.squeeze(out2.contiguous().cpu().numpy()).astype(np.float32)
    pred_conf = np.squeeze(out3.contiguous().cpu().numpy()).astype(np.float32)

    # opencv 2D visualization
    visualize_stereo_results_2d(left, right, pred_disp, pred_occ, pred_conf)


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)