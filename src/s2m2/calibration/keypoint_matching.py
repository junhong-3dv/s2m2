import numpy as np
from .base import evaluate_sample
import os
import sys
import cv2
import copy
from s2m2.core.utils.calib_utils import rotation_matrix_to_euler

def keypoint_based_calibration(left, right, calib_data):
    """
    Perform keypoint-matching based calibration to refine rotation matrix
    Args:
        left (np.ndarray): Raw Left image (grayscale)
        right (np.ndarray): Raw Right image (grayscale)
        calib_data (dict): Original calibration data
    """
    print("Starting keypoint-matching based online stereo calibration")

    # Convert to grayscale if needed
    if len(left.shape) == 3:
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left

    if len(right.shape) == 3:
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right

    # apply SIFT keypoint detector
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(left_gray, None)
    kp2, des2 = sift.detectAndCompute(right_gray, None)

    if des1 is None or des2 is None:
        print("Failed to detect keypoints in one or both images")
        return calib_data['stereo_extrinsic']['rotation']

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Good matches: {len(good)}")
    if len(good) < 10:
        print("Not enough good matches for calibration")
        return calib_data['stereo_extrinsic']['rotation']

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    # Use intrinsic parameters from XML calibration
    left_calib = calib_data['left']
    K = np.array([[left_calib['fx'], 0, left_calib['cx']],
                  [0, left_calib['fy'], left_calib['cy']],
                  [0, 0, 1]])

    print(f"Camera intrinsic matrix:\n{K}")

    # Estimate essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None:
        print("Failed to compute essential matrix")
        return calib_data['stereo_extrinsic']['rotation']

    print("Essential matrix computed successfully")

    # Extract R|t
    points, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # Get original rotation
    original_R = calib_data['stereo_extrinsic']['rotation']
    print("Original Rotation from XML calibration:")
    print(original_R)

    delta_R = R @ original_R.T
    roll_delta, pitch_delta, yaw_delta = rotation_matrix_to_euler(delta_R)
    print(f"Current deltas - Roll: {roll_delta:.4f}, Pitch: {pitch_delta:.4f}, Yaw: {yaw_delta:.4f}")

    calib_data_new = copy.deepcopy(calib_data)
    calib_data_new['stereo_extrinsic']['rotation'] = R

    return {
        'roll_delta': roll_delta,
        'pitch_delta': pitch_delta,
        'yaw_delta': yaw_delta,
        'calib_data_new': calib_data_new
    }
