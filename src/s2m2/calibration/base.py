import os
import sys

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import cv2
import torch
from s2m2.core.utils.model_utils import run_stereo_matching, compute_confidence_score
from s2m2.core.utils.image_utils import rectify_images
from s2m2.core.utils.vis_utils import apply_colormap, visualize_rectification
from s2m2.core.utils.calib_utils import compute_stereo_rectification, create_delta_rotation


def evaluate_sample(model, left, right, calib_data, device, roll_delta, pitch_delta, yaw_delta):
    """Evaluate a sample by computing confidence score"""
    # Get image size
    h, w = left.shape[:2]
    image_size = (w, h)
    try:
        # Create delta rotation
        delta_R = create_delta_rotation(roll_delta, pitch_delta, yaw_delta)
        rectification_data = compute_stereo_rectification(calib_data, image_size, delta_R)

        # Apply stereo rectification
        left_rectified, right_rectified = rectify_images(left, right, rectification_data)

        # Compute confidence score
        left_torch = (torch.from_numpy(left_rectified).permute(-1, 0, 1).unsqueeze(0)).to(device)
        right_torch = (torch.from_numpy(right_rectified).permute(-1, 0, 1).unsqueeze(0)).to(device)
        confidence_score = compute_confidence_score(model, left_torch, right_torch, device)

        return confidence_score if confidence_score is not None else 0.0
    except Exception as e:
        print(f"Error evaluating sample: {e}")
        return 0.0


def vivisualize_calibration_results(left, right, left_calibrated, right_calibrated, model, device):
    """Visualize CEM calibration results with rectification quality, disparity and confidence maps"""
    print("Computing stereo matching results for visualization...")

    # Run stereo matching on original images
    left_torch = (torch.from_numpy(left).permute(-1, 0, 1).unsqueeze(0)).to(device)
    right_torch = (torch.from_numpy(right).permute(-1, 0, 1).unsqueeze(0)).to(device)
    disp_before, occ_before, conf_before, avg_conf_score_before, _ = run_stereo_matching(model, left_torch, right_torch,
                                                                                         device)

    # Run stereo matching on calibrated images
    left_torch = (torch.from_numpy(left_calibrated).permute(-1, 0, 1).unsqueeze(0)).to(device)
    right_torch = (torch.from_numpy(right_calibrated).permute(-1, 0, 1).unsqueeze(0)).to(device)
    disp_after, occ_after, conf_after, avg_conf_score_after, _ = run_stereo_matching(model, left_torch, right_torch,
                                                                                     device)

    if disp_before is None or disp_after is None:
        print("Error: Failed to compute stereo matching results")
        return
    print(f"Initial confidence: {avg_conf_score_before:.4f}")
    print(f"Final confidence: {avg_conf_score_after:.4f}")
    # Convert to numpy arrays
    disp_before = disp_before.cpu().numpy()
    occ_before = occ_before.cpu().numpy()
    conf_before = conf_before.cpu().numpy()

    disp_after = disp_after.cpu().numpy()
    occ_after = occ_after.cpu().numpy()
    conf_after = conf_after.cpu().numpy()

    # Normalize disparity maps for visualization
    disp_before_vis = apply_colormap(disp_before)
    disp_after_vis = apply_colormap(disp_after)

    # Create comparison images
    # Original images with epipolar lines
    original_combined = visualize_rectification(left, right, num_lines=20, vis=False)

    # Calibrated images with epipolar lines
    calibrated_combined = visualize_rectification(left_calibrated, right_calibrated, num_lines=20, vis=False)

    # Disparity maps comparison
    disp_combined = np.hstack((disp_before_vis, disp_after_vis))

    # Confidence maps comparison
    conf_combined = np.hstack((conf_before, conf_after))

    # Show results
    cv2.namedWindow("Original Images with Epipolar Lines", cv2.WINDOW_NORMAL)
    cv2.imshow("Original Images with Epipolar Lines", original_combined)

    cv2.namedWindow("Calibrated Images with Epipolar Lines", cv2.WINDOW_NORMAL)
    cv2.imshow("Calibrated Images with Epipolar Lines", calibrated_combined)

    cv2.namedWindow("Disparity Maps (Before/After)", cv2.WINDOW_NORMAL)
    cv2.imshow("Disparity Maps (Before/After)", disp_combined)

    cv2.namedWindow("Confidence Maps (Before/After)", cv2.WINDOW_NORMAL)
    cv2.imshow("Confidence Maps (Before/After)", conf_combined)

    print("Press any key to close visualization windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
