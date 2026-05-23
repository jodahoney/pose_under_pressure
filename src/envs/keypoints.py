from __future__ import annotations

import numpy as np


DEFAULT_WALKER2D_BODIES = [
    "torso",
    "thigh",
    "leg",
    "foot",
    "thigh_left",
    "leg_left",
    "foot_left",
]


def get_mujoco_body_names(env) -> list[str]:
    """
    Return MuJoCo body names for a Gymnasium MuJoCo environment.
    """
    model = env.unwrapped.model
    return [model.body(i).name for i in range(model.nbody)]


def get_body_positions(env, body_names: list[str]) -> np.ndarray:
    """
    Return body center positions for selected MuJoCo bodies.

    Args:
        env: Gymnasium MuJoCo environment.
        body_names: Names of bodies to extract.

    Returns:
        positions: array of shape [K, 3].
    """
    model = env.unwrapped.model
    data = env.unwrapped.data

    positions = []
    for name in body_names:
        body_id = model.body(name).id
        positions.append(np.array(data.xpos[body_id], dtype=np.float32))

    return np.stack(positions, axis=0)


def root_relative_keypoints(
    positions: np.ndarray,
    root_index: int = 0,
    keep_dims: tuple[int, ...] = (0, 2),
) -> np.ndarray:
    """
    Convert absolute MuJoCo body positions into root-relative keypoints.

    MuJoCo positions are [x, y, z]. Walker2d is effectively planar, so we
    keep x and z by default.

    Args:
        positions: array of shape [K, 3].
        root_index: index of root body, usually torso.
        keep_dims: coordinate dimensions to keep.

    Returns:
        keypoints: array of shape [K, len(keep_dims)].
    """
    centered = positions - positions[root_index : root_index + 1]
    return centered[:, keep_dims].astype(np.float32)


def get_walker2d_keypoints(
    env,
    body_names: list[str] | None = None,
    root_relative: bool = True,
) -> np.ndarray:
    """
    Extract Walker2d body keypoints from the current simulator state.

    Returns:
        keypoints: array of shape [K, 2] if root_relative=True.
    """
    if body_names is None:
        body_names = DEFAULT_WALKER2D_BODIES

    positions = get_body_positions(env, body_names)

    if root_relative:
        return root_relative_keypoints(positions)

    return positions.astype(np.float32)
