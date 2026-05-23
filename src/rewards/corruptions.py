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
        Corrupted keypoints with same shape as input.
    """
    if sigma <= 0:
        return keypoints.astype(np.float32)

    noise = rng.normal(loc=0.0, scale=sigma, size=keypoints.shape)
    return (keypoints + noise).astype(np.float32)


def apply_keypoint_dropout(
    keypoints: np.ndarray,
    p_drop: float,
    rng: np.random.Generator,
    previous_keypoints: np.ndarray | None = None,
    mode: str = "hold",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply occlusion-style dropout to keypoints.

    Args:
        keypoints: array of shape [K, D].
        p_drop: probability of dropping each keypoint.
        rng: NumPy random generator.
        previous_keypoints: previous frame keypoints, used for mode='hold'.
        mode:
            'hold': replace dropped keypoints with previous-frame values.
            'zero': replace dropped keypoints with zeros.
            'mask': leave keypoints unchanged but return visibility mask.

    Returns:
        corrupted_keypoints: array of shape [K, D].
        visibility_mask: array of shape [K], where 1 means visible and 0 means dropped.
    """
    if p_drop <= 0:
        mask = np.ones(keypoints.shape[0], dtype=np.float32)
        return keypoints.astype(np.float32), mask

    if not 0 <= p_drop <= 1:
        raise ValueError(f"p_drop must be in [0, 1], got {p_drop}")

    visible = rng.random(keypoints.shape[0]) > p_drop
    mask = visible.astype(np.float32)

    corrupted = keypoints.copy().astype(np.float32)

    if mode == "hold":
        if previous_keypoints is None:
            replacement = keypoints
        else:
            replacement = previous_keypoints
        corrupted[~visible] = replacement[~visible]

    elif mode == "zero":
        corrupted[~visible] = 0.0

    elif mode == "mask":
        # For the raw dropout condition, this behaves like no coordinate change.
        # The mask is returned so masked rewards can be added later.
        pass

    else:
        raise ValueError(f"Unknown dropout mode: {mode}")

    return corrupted.astype(np.float32), mask
