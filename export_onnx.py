import os
import argparse
import numpy as np
import cv2
import torch
import onnxruntime

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from model.s2m2 import S2M2 as Model
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', default='S', type=str,
                        help='select model type: S,M,L,XL')
    parser.add_argument('--num_refine', default=1, type=int,
                        help='number of local iterative refinement')
    parser.add_argument('--allow_negative', action='store_true', help='allow negative disparity for imperfect rectification')
    parser.add_argument('--img_height', default=512, type=int,
                        help='image height')
    parser.add_argument('--img_width', default=512, type=int,
                        help='image width')

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
        print('model type should be one of [XS, S, M, L, XL]')
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

    img_height, img_width = args.img_height, args.img_width

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    model = load_model(args).to(device).eval()



    if args.allow_negative:
        left_path = 'samples/Web/64648_pbz98_3D_MPO_70pc_L.jpg'
        right_path = 'samples/Web/64648_pbz98_3D_MPO_70pc_R.jpg'
    else:
        left_path = 'samples/Web/0025_L.png'
        right_path = 'samples/Web/0025_R.png'

    # load stereo images
    left = cv2.cvtColor(cv2.imread(left_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(right_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


    img_height = (img_height // 32) * 32
    img_width = (img_width // 32) * 32
    print(f"image size: {img_height}, {img_width}")


    left = cv2.resize(left, dsize=(img_width, img_height))
    right = cv2.resize(right, dsize=(img_width, img_height))

    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).half().to(device)
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).half().to(device)

    torch_version = torch.__version__
    # onnx_path = f'S2M2_{args.model_type}_{img_width}_{img_height}_v2_torch{torch_version[0]}{torch_version[2]}.onnx'
    onnx_path = os.path.join('onnx_save', f'S2M2_{args.model_type}_{img_width}_{img_height}_v2_torch{torch_version[0]}{torch_version[2]}.onnx')

    print(onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)


    model = model.half().cpu()
    left_torch, right_torch = left_torch.half().cpu(), right_torch.half().cpu()

    print(f"ONNX conversion takes a long time, even for small model")
    try:
        torch.onnx.export(model,
                          (left_torch, right_torch),
                          onnx_path,
                          export_params=True,
                          opset_version=17,
                          verbose=False,
                          do_constant_folding=True,
                          input_names=['input_left', 'input_right'],
                          output_names=['output_disp', 'output_occ', 'output_conf'],
                          dynamic_axes=None)
        print('success onnx conversion')

    except Exception as e:
        print(f"error type:{type(e).__name__}")
        print(f"error :{e}")
        import traceback
        traceback.print_exc()

    # test onnx file with onnxruntime
    sess = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    print(f"onnxruntime device: {onnxruntime.get_device()}")

    input_name = [input.name for input in sess.get_inputs()]
    output_name = [output.name for output in sess.get_outputs()]
    print(f"input_name:{input_name}")
    print(f"output_name:{output_name}")

    outputs = sess.run([output_name[0], output_name[1], output_name[2]],
                          {input_name[0]: left_torch.numpy(),
                           input_name[1]: right_torch.numpy()})
    print(f"output shape: {outputs[1].shape}")

    pred_disp, pred_occ, pred_conf = outputs
    pred_disp = np.squeeze(pred_disp)
    pred_occ = np.squeeze(pred_occ)
    pred_conf = np.squeeze(pred_conf)


    # opencv 2D visualization
    valid = ((pred_conf >.1)*(pred_occ >.01))
    d_min = np.min(pred_disp)
    d_max = np.max(pred_disp)
    disp_left_vis = (pred_disp - d_min) / (d_max-d_min) * 255
    disp_left_vis = disp_left_vis.astype("uint8")
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
