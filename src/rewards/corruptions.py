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
    keypoints = keypoints.astype(np.float32)

    if sigma <= 0:
        return keypoints.copy()

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
    keypoints = keypoints.astype(np.float32)

    if not 0 <= p_drop <= 1:
        raise ValueError(f"p_drop must be in [0, 1], got {p_drop}")

    if mode not in {"hold", "zero", "mask"}:
        raise ValueError(f"Unknown dropout mode: {mode}")

    if p_drop <= 0:
        mask = np.ones(keypoints.shape[0], dtype=np.float32)
        return keypoints.copy(), mask

    visible = rng.random(keypoints.shape[0]) > p_drop
    mask = visible.astype(np.float32)

    corrupted = keypoints.copy()

    if mode == "hold":
        if previous_keypoints is None:
            replacement = keypoints
        else:
            replacement = previous_keypoints.astype(np.float32)

        if replacement.shape != keypoints.shape:
            raise ValueError(
                "previous_keypoints must have the same shape as keypoints, "
                f"got previous_keypoints={replacement.shape}, keypoints={keypoints.shape}"
            )

        corrupted[~visible] = replacement[~visible]

    elif mode == "zero":
        corrupted[~visible] = 0.0

    elif mode == "mask":
        # Leave coordinates unchanged. The returned mask is used by the reward
        # function to ignore dropped keypoints.
        pass

    return corrupted.astype(np.float32), mask


def add_systematic_bias(
    keypoints: np.ndarray,
    bias_indices: list[int] | np.ndarray,
    bias_magnitude: float,
    bias_axis: int = 0,
) -> np.ndarray:
    """
    Add a fixed offset to selected keypoints.

    Args:
        keypoints: array of shape [K, D].
        bias_indices: keypoint indices to corrupt.
        bias_magnitude: scalar offset applied to selected keypoints.
        bias_axis: coordinate axis to offset.

    Returns:
        Biased keypoints with same shape as input.
    """
    keypoints = keypoints.astype(np.float32)
    corrupted = keypoints.copy()

    if bias_magnitude == 0.0 or len(bias_indices) == 0:
        return corrupted

    if bias_axis < 0 or bias_axis >= corrupted.shape[1]:
        raise ValueError(
            f"bias_axis={bias_axis} is invalid for keypoint dimension {corrupted.shape[1]}"
        )

    corrupted[np.asarray(bias_indices, dtype=int), bias_axis] += bias_magnitude
    return corrupted.astype(np.float32)