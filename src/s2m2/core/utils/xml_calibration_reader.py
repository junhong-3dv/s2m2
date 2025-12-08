import xml.etree.ElementTree as ET
import numpy as np
import cv2

def parse_xml_calibration(xml_path):
    """
    Parse XML calibration file and extract all camera parameters
    
    Args:
        xml_path (str): Path to the XML calibration file
        
    Returns:
        dict: Dictionary containing all calibration parameters
    """
    tree = ET.parse(xml_path)
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

def load_calibration_from_xml(xml_path, image_size, delta_R=None):
    """
    Main function to load calibration data from XML and compute rectification
    
    Args:
        xml_path (str): Path to XML calibration file
        image_size (tuple): Image size (width, height)
        delta_R (np.ndarray, optional): Delta rotation matrix to apply to extrinsic R
        
    Returns:
        dict: Complete calibration and rectification data
    """
    # Parse XML calibration data
    calibration_data = parse_xml_calibration(xml_path)
    
    # Compute rectification parameters
    rectification_data = compute_stereo_rectification(calibration_data, image_size, delta_R)
    
    # Combine all data
    complete_data = {
        'calibration': calibration_data,
        'rectification': rectification_data
    }
    return complete_data
