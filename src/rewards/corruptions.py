from __future__ import annotations

import numpy as np


def add_gaussian_jitter(
    keypoints: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add independent zero-mean Gaussian noise to keypoints.

    Args:
        keypoints: array of shape [K, D].
        sigma: standard deviation of Gaussian noise.
        rng: NumPy random generator.

    Returns:
        corrupted keypoints with same shape as input.
    """
    if sigma <= 0:
        return keypoints

    noise = rng.normal(loc=0.0, scale=sigma, size=keypoints.shape)
    return (keypoints + noise).astype(np.float32)
