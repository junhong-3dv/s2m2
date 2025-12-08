# src/core/utils/vis_utils.py
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import math

def visualize_rectification(left_img, right_img, num_lines=20, vis=True):
    """
    Visualize rectification results with epipolar lines

    Args:
        left_img (np.ndarray): Rectified left image [H, W, 3 or 1]
        right_img (np.ndarray): Rectified right image [H, W, 3 or 1]
        num_lines (int): Number of epipolar lines to draw
    Returns:
        all_img (np.ndarray): left-right with horizontal lines
    """
    h, w = left_img.shape[:2]

    # Create side-by-side comparisons
    all_img = np.hstack((left_img, right_img))

    # Draw epipolar lines on rectified images
    for i in range(num_lines):
        y = int((i + 1) * h / (num_lines + 1))
        cv2.line(all_img, (0, y), (w * 2, y), (0, 255, 0), 1)

    if vis:
        # Display results
        cv2.imshow('Stereo Images with Epipolar Lines', all_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return all_img

def apply_colormap(x):
    x_norm = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return cv2.applyColorMap(x_norm, cv2.COLORMAP_JET)

def visualize_stereo_results_2d(left, right, pred_disp, pred_occ, pred_conf, vis=True):
    """
    Visualize stereo matching results as images

    Args:
        left (np.ndarray): Rectified left image [H, W, 3 or 1]
        right (np.ndarray): Rectified right image [H, W, 3 or 1]
        pred_disp (np.ndarray): predicted disparity map [H, W]
        pred_occ (np.ndarray): predicted occlusion map [H, W]
        pred_conf (np.ndarray): predicted confidence map [H, W]
        vis (bool): visualize the results in 2d

    Returns:
        disp_left_vis (np.ndarray): color coded disparity map
        disp_left_vis_mask (np.ndarray): color coded disparity map with confidence filtering
        valid (np.ndarray): trust region

    """

    valid = ((pred_conf >.1)*(pred_occ >.5))
    d_min = np.min(pred_disp)
    d_max = np.max(pred_disp)
    # disp_left_vis = (pred_disp - d_min) / (d_max-d_min) * 255
    # disp_left_vis = disp_left_vis.astype("uint8")
    # disp_left_vis = cv2.applyColorMap(disp_left_vis, cv2.COLORMAP_JET)
    pred_disp_vis = apply_colormap(pred_disp)
    pred_disp_vis_mask = valid[:,:,np.newaxis] * pred_disp_vis
    if vis:
        cv2.namedWindow('left-right', cv2.WINDOW_NORMAL)
        cv2.imshow('left-right', cv2.cvtColor(np.hstack((left, right)), cv2.COLOR_BGR2RGB))

        cv2.namedWindow(f'left_disparity: min:{round(d_min)}, max:{round(d_max)}', cv2.WINDOW_NORMAL)
        cv2.imshow(f'left_disparity: min:{round(d_min)}, max:{round(d_max)}', np.hstack((pred_disp_vis, pred_disp_vis_mask)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return pred_disp_vis, pred_disp_vis_mask, valid



def visualize_stereo_results_3d(pcd, pcd_filt=None):
    """
    Visualize stereo matching results as pointcloud

    Args:
        pcd (o3d.geometry.PointCloud): 3d pointcloud
        pcd_filt (o3d.geometry.PointCloud): 3d pointcloud with filtering
    """
    vis_transform = [[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]]  # transforms to make viewpoint match camera perspective
    pcd.transform(vis_transform)
    pcd_filt.transform(vis_transform)
    if pcd_filt is not None:
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