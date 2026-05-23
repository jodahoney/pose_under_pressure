from __future__ import annotations

import numpy as np


def pose_distance(
    agent_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Compute weighted squared keypoint distance.

    Args:
        agent_keypoints: array of shape [K, D].
        reference_keypoints: array of shape [K, D].
        weights: optional array of shape [K].

    Returns:
        Scalar squared distance.
    """
    diff = agent_keypoints - reference_keypoints
    sq_dist = np.sum(diff ** 2, axis=-1)

    if weights is not None:
        sq_dist = sq_dist * weights

    return float(np.sum(sq_dist))


def pose_reward(
    agent_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
    alpha: float = 5.0,
    weights: np.ndarray | None = None,
) -> float:
    """
    Exponential pose imitation reward.
    """
    dist = pose_distance(agent_keypoints, reference_keypoints, weights)
    return float(np.exp(-alpha * dist))
