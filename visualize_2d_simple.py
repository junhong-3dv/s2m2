import os
import argparse
import numpy as np
import cv2
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model.s2m2 import S2M2 as Model
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='S', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=3, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--torch_compile', action='store_true', help='apply torch_compile')
    parser.add_argument('--allow_negative', action='store_true', help='allow negative disparity for imperfect rectification')
    return parser

def load_model(args):

    if args.model_type == "S":
        feature_channels = 128
        n_transformer = 1 * 1
    elif args.model_type == "M":
        feature_channels = 192
        n_transformer = 1 * 2
    elif args.model_type == "L":
        feature_channels = 256
        n_transformer = 1 * 3
    elif args.model_type == "XL":
        feature_channels = 384
        n_transformer = 1*3
    else:
        print('model type should be one of [S, M, L, XL]')
        exit(1)


    model_path = 'CH' + str(feature_channels) + 'NTR' + str(n_transformer) + '.pth'
    ckpt_path = os.path.join('pretrain_weights', model_path)

    model = Model(feature_channels=feature_channels,
                  dim_expansion=1,
                  num_transformer=n_transformer,
                  use_positivity=not args.allow_negative,
                  refine_iter=args.num_refine
                  )
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.my_load_state_dict(checkpoint['state_dict'])
    return model

def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args).to(device).eval()
    if args.torch_compile:
        model = torch.compile(model)


    if args.allow_negative:
        left_path = 'samples/Web/64648_pbz98_3D_MPO_70pc_L.jpg'
        right_path = 'samples/Web/64648_pbz98_3D_MPO_70pc_R.jpg'
    else:
        left_path = 'samples/Web/0025_L.png'
        right_path = 'samples/Web/0025_R.png'

    # load stereo images
    left = cv2.cvtColor(cv2.imread(left_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(right_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    img_height, img_width = left.shape[:2]
    print(f"original image size: {img_height}, {img_width}")

    img_height = (img_height // 32) * 32
    img_width = (img_width // 32) * 32
    print(f"cropped image size: {img_height}, {img_width}")

    # image crop
    left = left[:img_height, :img_width]
    right = right[:img_height, :img_width]

    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).half().to(device)
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).half().to(device)

    # from fvcore.nn import FlopCountAnalysis, flop_count_table
    # # print(f"flops: {flops.total()}")
    # with torch.inference_mode():
    #     with torch.amp.autocast(enabled=True, device_type='cuda', dtype=torch.float16):
    #         flops = FlopCountAnalysis(model, (left_torch, right_torch))
    #         print(f"Gflops: {flops.total() / 1e9}")
    # print(flop_count_table(flops))
    #


    with torch.no_grad():
        with torch.amp.autocast(enabled=True, device_type=device.type, dtype=torch.float16):
            print(f"pre-run...")
            _ = model(left_torch, right_torch)
            T = 1
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(T):
                pred_disp, pred_occ, pred_conf = model(left_torch, right_torch)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

    print(F"torch avg inference time:{(curr_time)/T/1000}, FPS:{1000*T/(curr_time)}")

    # opencv 2D visualization
    valid = ((pred_conf.cpu().float() >.1)).squeeze().numpy()
    d_min = pred_disp.min().item()
    d_max = pred_disp.max().item()
    disp_left_vis = (pred_disp - d_min) / (d_max-d_min) * 255
    disp_left_vis = disp_left_vis.cpu().numpy()[0,0].astype("uint8")
    disp_left_vis = cv2.applyColorMap(disp_left_vis, cv2.COLORMAP_JET)
    disp_left_vis_mask = valid[:,:,np.newaxis] * disp_left_vis

    cv2.imshow('left-right', np.hstack((left, right)))
    cv2.imshow(f'left_disparity: min:{round(d_min)}, max:{round(d_max)}', np.hstack((disp_left_vis,disp_left_vis_mask)))
    cv2.waitKey(0)


if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()
    print(args)
    main(args)
