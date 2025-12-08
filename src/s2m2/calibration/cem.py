import numpy as np
from .base import evaluate_sample
import copy
import os
import sys
from s2m2.core.utils.calib_utils import euler_to_rotation_matrix, apply_delta_rotation


def cem_calibration(model, left, right, calib_data, device, **kwargs):
    """
    Perform CEM based calibration to refine rotation matrix
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
        'num_samples': 20,
        'num_elite': 3,
        'initial_std': 0.002,
        'std_decay': 0.8}
    config.update(kwargs)
    print("Starting CEM based online stereo calibration")


    max_iterations, num_samples, num_elite = config['max_iterations'], config['num_samples'], config['num_elite']
    initial_std, initial_std, initial_std = config['initial_std'], config['initial_std'], config['initial_std']
    std_decay = config['std_decay']

    # Validate parameters
    if num_elite > num_samples:
        print(f"Warning: num_elite ({num_elite}) cannot be greater than num_samples ({num_samples})")
        print("Setting num_elite to num_samples")
        num_elite = num_samples

    # Get initial confidence
    initial_confidence = evaluate_sample(model, left, right, calib_data, device, 0, 0, 0)
    print(f"Initial confidence: {initial_confidence:.4f}")

    # Initialize parameters
    # Mean for [roll, pitch, yaw]
    mean_params = np.array([0.0, 0.0, 0.0])
    std_params = np.array([initial_std, initial_std, initial_std])

    current_confidence = initial_confidence
    best_params = mean_params.copy()
    best_confidence = initial_confidence

    # CEM iterations
    for iteration in range(max_iterations):
        if best_confidence > 0.98:
            break;
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        print(f"Current confidence: {current_confidence:.4f}")
        print(f"Current mean - Roll: {mean_params[0]:.4f}, Pitch: {mean_params[1]:.4f}, Yaw: {mean_params[2]:.4f}")
        print(f"Current std - Roll: {std_params[0]:.4f}, Pitch: {std_params[1]:.4f}, Yaw: {std_params[2]:.4f}")

        # Generate samples
        samples = np.random.normal(mean_params, std_params, (num_samples, 3))

        # Evaluate samples
        sample_scores = []
        sample_scores.append((mean_params, current_confidence))
        for i in range(num_samples):
            roll, pitch, yaw = samples[i]
            confidence = evaluate_sample(model, left, right, calib_data, device, roll, pitch, yaw)
            sample_scores.append((samples[i], confidence))

        # Sort by confidence score (descending)
        sample_scores.sort(key=lambda x: x[1], reverse=True)

        # Select elite samples
        elite_samples = [sample for sample, score in sample_scores[:num_elite]]
        elite_scores = [score for sample, score in sample_scores[:num_elite]]

        # Update mean and std
        elite_samples = np.array(elite_samples)
        mean_params = np.mean(elite_samples, axis=0)
        std_params = np.std(elite_samples, axis=0) * std_decay  # Apply decay

        # Ensure minimum std to avoid collapse
        std_params = np.maximum(std_params, 0.00005)

        # Update best parameters if needed
        if elite_scores[0] > best_confidence:
            best_confidence = elite_scores[0]
            best_params = elite_samples[0].copy()
            # Update current confidence
            current_confidence = elite_scores[0]

        print(f"Best sample confidence: {elite_scores[0]:.4f}")
        print(f"Elite mean - Roll: {mean_params[0]:.4f}, Pitch: {mean_params[1]:.4f}, Yaw: {mean_params[2]:.4f}")

    # Final results
    print("\n" + "=" * 50)
    print("CEM CALIBRATION RESULTS")
    print("=" * 50)
    print(f"Initial confidence: {initial_confidence:.4f}")
    print(f"Final confidence: {best_confidence:.4f}")
    print(f"Confidence improvement: {best_confidence - initial_confidence:+.4f}")
    print(f"Final deltas - Roll: {best_params[0]:.4f}, Pitch: {best_params[1]:.4f}, Yaw: {best_params[2]:.4f}")


    calib_data_new = copy.deepcopy(calib_data)
    R = apply_delta_rotation(calib_data['stereo_extrinsic']['rotation'],
                              euler_to_rotation_matrix(best_params[0],best_params[1],best_params[2]))
    calib_data_new['stereo_extrinsic']['rotation'] = R

    return {
        'roll_delta': best_params[0],
        'pitch_delta': best_params[1],
        'yaw_delta': best_params[2],
        'initial_confidence': initial_confidence,
        'final_confidence': best_confidence,
        'calib_data_new': calib_data_new
    }
