import numpy as np
from .base import evaluate_sample
import copy
import os
import sys
from s2m2.core.utils.calib_utils import euler_to_rotation_matrix, apply_delta_rotation

def gradient_descent_calibration(model, left, right, calib_data, device, **kwargs):
    """
    Perform gradient-descent based calibration to refine rotation matrix
    Args:
        model (np.ndarray): Stereo Matching Model
        left (np.ndarray): Raw Left image (grayscale)
        right (np.ndarray): Raw Right image (grayscale)
        calib_data (dict): Original calibration data
        device : torch.device
    """

    # default settings
    config = {
        'max_iterations': 5,
        'step_size': 0.0001,
        'eps': 0.01}
    config.update(kwargs)

    print("Starting gradient descent based online stereo calibration with line search")

    max_iterations, step_size, eps = config['max_iterations'], config['step_size'], config['eps']

    # Get initial confidence
    initial_confidence = evaluate_sample(model, left, right, calib_data, device, 0, 0, 0)
    print(f"Initial confidence: {initial_confidence:.4f}")

    # Initialize parameters
    # Mean for [roll, pitch, yaw]
    mean_params = np.array([0.0, 0.0, 0.0])

    current_confidence = initial_confidence

    # GD iterations
    roll_delta, pitch_delta, yaw_delta = mean_params.copy()
    for iteration in range(max_iterations):
        if current_confidence > 0.98:
            break;
        print(f"\nIteration {iteration + 1}/{max_iterations}")

        # Compute gradient and update for roll axis
        roll_delta, pitch_delta, yaw_delta, current_confidence = compute_gradient_update(
            model, left, right, calib_data, device, roll_delta, pitch_delta, yaw_delta, 'roll', eps, step_size)
        print(f"Current confidence: {current_confidence:.4f}")
        print(f"Current deltas - Roll: {roll_delta:.4f}, Pitch: {pitch_delta:.4f}, Yaw: {yaw_delta:.4f}")


        roll_delta, pitch_delta, yaw_delta, current_confidence = compute_gradient_update(
            model, left, right, calib_data, device, roll_delta, pitch_delta, yaw_delta, 'pitch', eps, step_size)
        print(f"Current confidence: {current_confidence:.4f}")
        print(f"Current deltas - Roll: {roll_delta:.4f}, Pitch: {pitch_delta:.4f}, Yaw: {yaw_delta:.4f}")

        roll_delta, pitch_delta, yaw_delta, current_confidence = compute_gradient_update(
            model, left, right, calib_data, device, roll_delta, pitch_delta, yaw_delta, 'yaw', eps, step_size)
        print(f"Current confidence: {current_confidence:.4f}")
        print(f"Current deltas - Roll: {roll_delta:.4f}, Pitch: {pitch_delta:.4f}, Yaw: {yaw_delta:.4f}")

    # Final results
    print("\n" + "=" * 50)
    print("CEM CALIBRATION RESULTS")
    print("=" * 50)
    print(f"Initial confidence: {initial_confidence:.4f}")
    print(f"Final confidence: {current_confidence:.4f}")
    print(f"Confidence improvement: {current_confidence - initial_confidence:+.4f}")
    print(f"Final deltas - Roll: {roll_delta:.4f}, Pitch: {pitch_delta:.4f}, Yaw: {yaw_delta:.4f}")

    calib_data_new = copy.deepcopy(calib_data)
    calib_data_new['stereo_extrinsic']['rotation'] = euler_to_rotation_matrix(roll_delta,pitch_delta,yaw_delta)

    calib_data_new = copy.deepcopy(calib_data)
    R = apply_delta_rotation(calib_data['stereo_extrinsic']['rotation'],
                              euler_to_rotation_matrix(roll_delta,pitch_delta,yaw_delta))
    calib_data_new['stereo_extrinsic']['rotation'] = R

    return {
        'roll_delta': roll_delta,
        'pitch_delta': pitch_delta,
        'yaw_delta': yaw_delta,
        'initial_confidence': initial_confidence,
        'final_confidence': current_confidence
    }

def compute_gradient_update(model, left, right, calib_data, device, roll, pitch, yaw, axis, eps, stepsize, max_searches=5):
    """Compute gradient for a specific rotation axis (roll, pitch, yaw)"""


    current_confidence = evaluate_sample(model, left, right, calib_data, device, roll, pitch, yaw)
    if axis == 'roll':
        eps_confidence = evaluate_sample(model, left, right, calib_data, device, roll + eps, pitch, yaw)
    elif axis == 'pitch':
        eps_confidence = evaluate_sample(model, left, right, calib_data, device, roll, pitch + eps, yaw)
    elif axis == 'yaw':
        eps_confidence = evaluate_sample(model, left, right, calib_data, device, roll, pitch, yaw + eps)

    # Compute gradient (derivative)
    gradient = (eps_confidence - current_confidence) / eps

    # Check for NaN or invalid values
    if np.isnan(gradient) or np.isinf(gradient):
        print(f"Warning: Invalid gradient for {axis}, setting to 0.0")
        return 0.0

    # Simple Line search for optimal step size

    if abs(gradient) > 1e-6:  # Only update if gradient is significant
        best_stepsize = 0.0
        best_confidence = current_confidence
        for i in range(max_searches):
            # Apply update with current stepsize
            if axis == 'roll':
                roll_new = roll + stepsize * gradient
                pitch_new = pitch
                yaw_new = yaw
            elif axis == 'pitch':
                roll_new = roll
                pitch_new = pitch + stepsize * gradient
                yaw_new = yaw
            elif axis == 'yaw':
                roll_new = roll
                pitch_new = pitch
                yaw_new = yaw + stepsize * gradient

            # Compute new confidence
            try:
                new_confidence = evaluate_sample(model, left, right, calib_data, device, roll_new, pitch_new, yaw_new)
                if new_confidence is not None and new_confidence > current_confidence:
                    # Found improvement, use this stepsize
                    best_stepsize = stepsize
                    best_confidence = new_confidence
                    print(f"    Found improvement at step {i + 1}: {current_confidence:.4f} -> {new_confidence:.4f}")
                    break
                else:
                    if new_confidence is not None:
                        print(f"    No improvement at step {i + 1}: {current_confidence:.4f} -> {new_confidence:.4f}")
                    else:
                        print(f"    No improvement at step {i + 1}: {current_confidence:.4f} -> None")
            except Exception as e:
                print(f"    Error at step {i + 1}: {str(e)}")

            # Reduce stepsize
            stepsize *= 0.25

        if best_stepsize == 0.0:
            print(f"  No improvement found for {axis}, skipping update")
        else:
            print(f"  Using stepsize {best_stepsize} for {axis}")

    else:
        print("  Skipping roll update due to small gradient")

    if axis == 'roll':
        roll_new = roll + best_stepsize * gradient
        pitch_new = pitch
        yaw_new = yaw
    elif axis == 'pitch':
        roll_new = roll
        pitch_new = pitch + best_stepsize * gradient
        yaw_new = yaw
    elif axis == 'yaw':
        roll_new = roll
        pitch_new = pitch
        yaw_new = yaw + best_stepsize * gradient

    return roll_new, pitch_new, yaw_new, best_confidence
