# src/core/utils/calib_utils.py
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import os 
from .xml_calibration_reader import load_calibration_from_xml

def load_calibration_data(calib_xml_path):
    """load calibration data"""

    if not os.path.exists(calib_xml_path):
        print(f"XML calibration file not found: {calib_xml_path}")
        return None
    try:
        calib_data = parse_xml_calibration(calib_xml_path)
        print("Calibration data loaded")

        return calib_data
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        return None



def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to rotation matrix

    Args:
        roll (float): Rotation around x-axis (radians)
        pitch (float): Rotation around y-axis (radians)
        yaw (float): Rotation around z-axis (radians)

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Create rotation object from Euler angles
    r = R.from_euler('xyz', [roll, pitch, yaw])
    return r.as_matrix()


def axis_angle_to_rotation_matrix(axis, angle):
    """
    Convert axis-angle representation to rotation matrix

    Args:
        axis (np.ndarray): 3D rotation axis (should be normalized)
        angle (float): Rotation angle in radians

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    # Create rotation object from axis-angle
    r = R.from_rotvec(axis * angle)
    return r.as_matrix()


def rotation_matrix_to_euler(rot_matrix):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw)

    Args:
        rot_matrix (np.ndarray): 3x3 rotation matrix

    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Create rotation object from matrix
    r = R.from_matrix(rot_matrix)
    return r.as_euler('xyz')


def create_delta_rotation(roll_delta=0.0, pitch_delta=0.0, yaw_delta=0.0):
    """
    Create a delta rotation matrix from small angle changes

    Args:
        roll_delta (float): Roll change in radians
        pitch_delta (float): Pitch change in radians
        yaw_delta (float): Yaw change in radians

    Returns:
        np.ndarray: 3x3 delta rotation matrix
    """
    return euler_to_rotation_matrix(roll_delta, pitch_delta, yaw_delta)


def apply_delta_rotation(original_R, delta_R):
    """
    Apply delta rotation to original rotation matrix

    Args:
        original_R (np.ndarray): Original 3x3 rotation matrix
        delta_R (np.ndarray): Delta 3x3 rotation matrix

    Returns:
        np.ndarray: New 3x3 rotation matrix
    """
    return original_R @ delta_R


def validate_rotation_matrix(R):
    """
    Validate if a matrix is a valid rotation matrix

    Args:
        R (np.ndarray): 3x3 matrix to validate

    Returns:
        bool: True if valid rotation matrix, False otherwise
    """
    # Check if matrix is 3x3
    if R.shape != (3, 3):
        return False

    # Check if matrix is orthogonal (R * R.T = I)
    identity = np.eye(3)
    is_orthogonal = np.allclose(R @ R.T, identity, atol=1e-6)

    # Check if determinant is 1
    has_unit_determinant = np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

    return is_orthogonal and has_unit_determinant


def small_angle_rotation_to_matrix(delta_angles):
    """
    Convert small angle rotations to rotation matrix using small angle approximation

    For small angles, this is a computationally efficient approximation

    Args:
        delta_angles (np.ndarray): [delta_roll, delta_pitch, delta_yaw] in radians

    Returns:
        np.ndarray: 3x3 delta rotation matrix
    """
    delta_roll, delta_pitch, delta_yaw = delta_angles

    # Small angle approximation rotation matrix
    # For small angles, sin(θ) ≈ θ and cos(θ) ≈ 1
    R = np.array([
        [1.0, -delta_yaw, delta_pitch],
        [delta_yaw, 1.0, -delta_roll],
        [-delta_pitch, delta_roll, 1.0]
    ])

    return R


# Example usage and testing functions
def test_delta_rotation():
    """
    Test function to demonstrate delta rotation usage
    """
    # Create a small delta rotation (in radians)
    delta_R = create_delta_rotation(
        roll_delta=0.01,  # ~0.57 degrees
        pitch_delta=0.005,  # ~0.29 degrees
        yaw_delta=0.002  # ~0.11 degrees
    )

    print("Delta rotation matrix:")
    print(delta_R)
    print(f"Is valid rotation matrix: {validate_rotation_matrix(delta_R)}")

    # Example of applying to identity matrix
    original_R = np.eye(3)
    new_R = apply_delta_rotation(original_R, delta_R)

    print("\nOriginal R (identity):")
    print(original_R)
    print("\nNew R after applying delta:")
    print(new_R)

    return delta_R


import xml.etree.ElementTree as ET
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R


def parse_xml_calibration(calib_xml_path):
    """
    Parse XML calibration file and extract all camera parameters

    Args:
        calib_xml_path (str): Path to the XML calibration file

    Returns:
        dict: Dictionary containing all calibration parameters
    """
    tree = ET.parse(calib_xml_path)
    root = tree.getroot()

    calibration_data = {}

    # Parse left camera intrinsic parameters
    left_intrinsic = root.find('distorted_left_intrinsic')
    calibration_data['left'] = {
        'fx': float(left_intrinsic.find('fx').text),
        'fy': float(left_intrinsic.find('fy').text),
        'cx': float(left_intrinsic.find('cx').text),
        'cy': float(left_intrinsic.find('cy').text),
        'distortion': parse_distortion_coeffs(left_intrinsic.find('dist').text)
    }

    # Parse right camera intrinsic parameters
    right_intrinsic = root.find('distorted_right_intrinsic')
    calibration_data['right'] = {
        'fx': float(right_intrinsic.find('fx').text),
        'fy': float(right_intrinsic.find('fy').text),
        'cx': float(right_intrinsic.find('cx').text),
        'cy': float(right_intrinsic.find('cy').text),
        'distortion': parse_distortion_coeffs(right_intrinsic.find('dist').text)
    }

    # Parse RGB camera intrinsic parameters (if needed)
    rgb_intrinsic = root.find('distorted_rgb_intrinsic')
    calibration_data['rgb'] = {
        'fx': float(rgb_intrinsic.find('fx').text),
        'fy': float(rgb_intrinsic.find('fy').text),
        'cx': float(rgb_intrinsic.find('cx').text),
        'cy': float(rgb_intrinsic.find('cy').text),
        'distortion': parse_distortion_coeffs(rgb_intrinsic.find('dist').text)
    }

    # Parse stereo extrinsic parameters
    stereo_extrinsic = root.find('stereo_extrinsic')
    calibration_data['stereo_extrinsic'] = {
        'rotation': parse_rotation_matrix(stereo_extrinsic.find('rotation').text),
        'translation': parse_translation_vector(stereo_extrinsic.find('translation').text)
    }

    # Parse left to RGB transformation (if needed)
    left2rgb = root.find('left2rgb')
    calibration_data['left2rgb'] = {
        'rotation': parse_rotation_matrix(left2rgb.find('rotation').text),
        'translation': parse_translation_vector(left2rgb.find('translation').text)
    }

    return calibration_data


def parse_distortion_coeffs(dist_str):
    """
    Parse distortion coefficient string into numpy array

    Args:
        dist_str (str): Comma-separated distortion coefficients

    Returns:
        np.ndarray: Distortion coefficients array
    """
    coeffs = [float(x.strip()) for x in dist_str.split(',')]
    return np.array(coeffs)


def parse_rotation_matrix(rot_str):
    """
    Parse rotation matrix string into 3x3 numpy array

    Args:
        rot_str (str): Comma-separated rotation matrix elements

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    elements = [float(x.strip()) for x in rot_str.split(',')]
    return np.array(elements).reshape(3, 3)


def parse_translation_vector(trans_str):
    """
    Parse translation vector string into numpy array

    Args:
        trans_str (str): Comma-separated translation vector elements

    Returns:
        np.ndarray: Translation vector
    """
    elements = [float(x.strip()) for x in trans_str.split(',')]
    return np.array(elements)


def build_camera_matrix(fx, fy, cx, cy):
    """
    Build camera intrinsic matrix K

    Args:
        fx (float): Focal length in x direction
        fy (float): Focal length in y direction
        cx (float): Principal point x coordinate
        cy (float): Principal point y coordinate

    Returns:
        np.ndarray: 3x3 camera intrinsic matrix
    """
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]])


def compute_stereo_rectification(calibration_data, image_size, delta_R=None):
    """
    Compute stereo rectification parameters using OpenCV

    Args:
        calibration_data (dict): Calibration data from XML
        image_size (tuple): Image size (width, height)
        delta_R (np.ndarray, optional): Delta rotation matrix to apply to extrinsic R

    Returns:
        dict: Rectification parameters including R1, R2, P1, P2, Q, and maps
    """
    # Extract camera matrices
    K1 = build_camera_matrix(
        calibration_data['left']['fx'],
        calibration_data['left']['fy'],
        calibration_data['left']['cx'],
        calibration_data['left']['cy']
    )

    K2 = build_camera_matrix(
        calibration_data['right']['fx'],
        calibration_data['right']['fy'],
        calibration_data['right']['cx'],
        calibration_data['right']['cy']
    )

    # Extract distortion coefficients
    D1 = calibration_data['left']['distortion']
    D2 = calibration_data['right']['distortion']

    # Extract extrinsic parameters
    R = calibration_data['stereo_extrinsic']['rotation']
    T = calibration_data['stereo_extrinsic']['translation']

    # Apply delta rotation if provided
    if delta_R is not None:
        R = R @ delta_R

    # Compute rectification transforms
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0  # Set to 0 to crop all invalid pixels
    )

    # Compute rectification maps
    leftMapX, leftMapY = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1
    )
    rightMapX, rightMapY = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_32FC1
    )

    return {
        'K1': K1, 'D1': D1, 'K2': K2, 'D2': D2,
        'R': R, 'T': T, 'R1': R1, 'R2': R2,
        'P1': P1, 'P2': P2, 'Q': Q,
        'leftMapX': leftMapX, 'leftMapY': leftMapY,
        'rightMapX': rightMapX, 'rightMapY': rightMapY
    }


