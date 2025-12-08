# src/core/utils/image_utils.py
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import math

def read_images(left_img_path, right_img_path):
    """
    load stereo images

    Args:
        left_img_path: left image path
        right_img_path: right image path

    Returns:
        left_img: left image
        right_img: right image
    """

    left = cv2.cvtColor(cv2.imread(left_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(right_img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return left, right


@torch.no_grad()
def image_pad(img, factor=32):
    """
    zero-pad image with divisive factor

    Args:
        img (torch.Tensor): image [B, C, H, W]
        factor (int): divisive factor

    Returns:
        img_crop: cropped image
    """
    H,W = img.shape[-2:]

    # get divisive factor
    H_new = math.ceil(H / factor) * factor
    W_new = math.ceil(W / factor) * factor

    pad_h = H_new - H
    pad_w = W_new - W

    # horizontal padding
    p2d = (pad_w//2, pad_w-pad_w//2, 0, 0)
    img_pad = F.pad(img, p2d, "constant", 0)
    # vertical padding
    p2d = (0,0, pad_h // 2, pad_h - pad_h // 2)
    img_pad = F.pad(img_pad, p2d, "constant", 0)

    img_pad_down = F.adaptive_avg_pool2d(img_pad.float(), output_size=[H // factor, W // factor])
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

@torch.no_grad()
def image_crop(img, img_shape):
    """
    crop image with shape

    Args:
        img (torch.Tensor): image [... H, W]
        img_shape (list): image shape to crop

    Returns:
        img_crop: cropped image
    """

    H, W = img.shape[-2:]
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

    img_crop = img

    return img_crop




def rectify_images(left_img, right_img, rectification_data):
    """
    Apply stereo rectification to left and right images

    Args:
        left_img (np.ndarray): Left image
        right_img (np.ndarray): Right image
        rectification_data (dict): Rectification parameters from compute_stereo_rectification

    Returns:
        tuple: Rectified left and right images
    """
    left_rectified = cv2.remap(
        left_img,
        rectification_data['leftMapX'],
        rectification_data['leftMapY'],
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT
    )

    right_rectified = cv2.remap(
        right_img,
        rectification_data['rightMapX'],
        rectification_data['rightMapY'],
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT
    )

    return left_rectified, right_rectified

