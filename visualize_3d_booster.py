import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
# import re
import cv2
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.nn.functional as F
from model.s2m2 import S2M2 as Model
import open3d as o3d
import argparse
import math


device='cuda'
import torch._dynamo
torch._dynamo.config.verbose=True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='XL', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=3, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--torch_compile', action='store_true', help='torch_compile')
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
                  use_positivity=True,
                  refine_iter=args.num_refine
                  )
    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.my_load_state_dict(checkpoint['state_dict'])
    return model



def get_pointcloud(rgb, disp, calib):
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
                                                              depth_trunc=1000.0,
                                                              convert_rgb_to_intensity=False)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fx, cx, cy)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return point_cloud

def read_calib_file(calib_path):


    calib = {}
    file = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    calib['cam0'] = file.getNode("proj_matL").mat()
    calib['baseline'] = float(file.getNode("baselineLR").real()) * 1.
    calib['doffs'] = 0

    return calib

def image_pad(img, factor):
    with torch.no_grad():
        H,W = img.shape[-2:]

        H_new = math.ceil(H / factor) * factor
        W_new = math.ceil(W / factor) * factor

        pad_h = H_new - H
        pad_w = W_new - W

        p2d = (pad_w//2, pad_w-pad_w//2, 0, 0)
        img_pad = F.pad(img, p2d, "constant", 0)
        #
        p2d = (0,0, pad_h // 2, pad_h - pad_h // 2)
        img_pad = F.pad(img_pad, p2d, "constant", 0)

        img_pad_down = F.adaptive_avg_pool2d(img_pad, output_size=[H // factor, W // factor])
        img_pad = F.interpolate(img_pad_down, size=[H_new, W_new], mode='bilinear')

        h_s = pad_h // 2
        h_e = (pad_h - pad_h // 2)
        w_s = pad_w // 2
        w_e = (pad_w - pad_w // 2)
        if h_e==0 and w_e==0:
            img_pad[:, :, h_s:, w_s:] = img
        elif h_e==0:
            img_pad[:, :, h_s:, w_s:-w_e] = img
        elif w_e==0:
            img_pad[:, :, h_s:-h_e, w_s:] = img
        else:
            img_pad[:, :, h_s:-h_e, w_s:-w_e] = img

        return img_pad

def image_crop(img, img_shape):
    with torch.no_grad():
        H,W = img.shape[-2:]
        H_new, W_new = img_shape

        crop_h = H - H_new
        if crop_h > 0:
            crop_s = crop_h // 2
            crop_e = crop_h - crop_h // 2
            img = img[:,:,crop_s: -crop_e]

        crop_w = W - W_new
        if crop_w > 0:
            crop_s = crop_w // 2
            crop_e = crop_w - crop_w // 2
            img = img[:,:,:, crop_s: -crop_e]

        return img


def inference(imgL, imgR, model, args):

    print('processing...')
    with torch.no_grad():
        with torch.amp.autocast(enabled=True, device_type='cuda:0', dtype=torch.float16):
            result = model(imgL, imgR, refine_iter=args.num_refine)

    disp_est = result['flow_preds']
    pred_conf = result['conf_preds']
    occ_est = result['occ_preds']

    return disp_est[-1], occ_est[-1], pred_conf[-1]


def main(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args).to(device).eval()
    if args.torch_compile:
        model = torch.compile(model)


    left_path = 'samples/Lid/im0.png'
    right_path = 'samples/Lid/im1.png'
    calib_path = 'samples/Lid/calib.xml'

    # load stereo images
    left = cv2.cvtColor(cv2.imread(left_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(right_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    left = cv2.resize(left,(0,0), fx=1/2, fy=1/2)
    right = cv2.resize(right,(0,0), fx=1/2, fy=1/2)

    # load calibration params
    calib = read_calib_file(calib_path)

    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).half().to(device)
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).half().to(device)

    left_torch_pad = image_pad(left_torch, 32)
    right_torch_pad = image_pad(right_torch, 32)


    img_height, img_width = left.shape[:2]
    print(f"original image size: {img_height}, {img_width}")

    img_height_pad, img_width_pad = left_torch_pad.shape[2:]
    print(f"padded image size: {img_height_pad}, {img_width_pad}")

    with torch.no_grad():
        with torch.amp.autocast(enabled=True, device_type=device.type, dtype=torch.float16):
            print(f"pre-run...")
            _ = model(left_torch_pad, right_torch_pad)
            T = 1
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(T):
                pred_disp, pred_occ, pred_conf = model(left_torch_pad, right_torch_pad)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

    print(F"torch avg inference time:{(curr_time)/T/1000}, FPS:{1000*T/(curr_time)}")

    pred_disp = image_crop(pred_disp, (img_height, img_width))
    pred_occ = image_crop(pred_occ, (img_height, img_width))
    pred_conf = image_crop(pred_conf, (img_height, img_width))


    # opencv 2D visualization
    valid = (((pred_conf).cpu().float() >.1)*((pred_occ).cpu().float() >.01)).squeeze().numpy()
    d_min = pred_disp.min().item()
    d_max = pred_disp.max().item()
    disp_left_vis = (pred_disp - d_min) / (d_max-d_min) * 255
    disp_left_vis = disp_left_vis.cpu().squeeze().numpy().astype("uint8")
    disp_left_vis = cv2.applyColorMap(disp_left_vis, cv2.COLORMAP_JET)
    disp_left_vis_mask = valid[:,:,np.newaxis] * disp_left_vis


    # open3d pointcloud visualization
    pred_disp_np = np.ascontiguousarray(pred_disp.squeeze().cpu().float().numpy()).astype(np.float32)
    pred_disp_np_filt = pred_disp_np * valid
    pred_disp_np_filt[~valid] = -1

    vis_transform = [[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]]  # transforms to make viewpoint match camera perspective
    pcd = get_pointcloud(left, pred_disp_np, calib)
    pcd.transform(vis_transform)

    pcd_filt = get_pointcloud(left, pred_disp_np_filt, calib)
    pcd_filt.transform(vis_transform)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pointcloud with confidence filtering')
    render_option = vis.get_render_option()
    render_option.point_size=1.0
    # render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    render_option.point_color_option = o3d.visualization.PointColorOption.Color
    vis.add_geometry(pcd_filt)
    vis.run()
    vis.destroy_window()
    vis.create_window(window_name='pointcloud without filtering')
    render_option = vis.get_render_option()
    render_option.point_size=1.0
    # render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    render_option.point_color_option = o3d.visualization.PointColorOption.Color
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()



    cv2.namedWindow('left-right', cv2.WINDOW_NORMAL)
    cv2.imshow('left-right', np.hstack((left, right)))
    cv2.namedWindow(f'left_disparity: min:{round(d_min)}, max:{round(d_max)}', cv2.WINDOW_NORMAL)
    cv2.imshow(f'left_disparity: min:{round(d_min)}, max:{round(d_max)}', np.hstack((disp_left_vis,disp_left_vis_mask)))
    cv2.waitKey(0)

if __name__ == '__main__':

    parser = get_args_parser()
    args = parser.parse_args()

    main(args)



