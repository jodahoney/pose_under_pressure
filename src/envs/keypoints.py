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

DEFAULT_HOPPER_BODIES = [
    "torso",
    "thigh",
    "leg",
    "foot",
]


def get_env_id(env) -> str:
    """
    Return the Gymnasium environment id when available.
    """
    spec = getattr(env.unwrapped, "spec", None)
    if spec is not None and getattr(spec, "id", None) is not None:
        return spec.id
    return env.unwrapped.__class__.__name__


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

    available = set(get_mujoco_body_names(env))
    missing = [name for name in body_names if name not in available]
    if missing:
        raise ValueError(
            f"Requested body names are missing from {get_env_id(env)}: {missing}. "
            f"Available bodies: {sorted(available)}"
        )

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

    MuJoCo positions are [x, y, z]. Walker2d and Hopper are effectively
    planar, so we keep x and z by default.

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

    Default keypoint order:
        0 torso
        1 thigh
        2 leg
        3 foot
        4 thigh_left
        5 leg_left
        6 foot_left

    Returns:
        keypoints: array of shape [7, 2] if root_relative=True.
    """
    if body_names is None:
        body_names = DEFAULT_WALKER2D_BODIES

    positions = get_body_positions(env, body_names)

    if root_relative:
        return root_relative_keypoints(positions)

    return positions.astype(np.float32)


def get_hopper_keypoints(
    env,
    body_names: list[str] | None = None,
    root_relative: bool = True,
) -> np.ndarray:
    """
    Extract Hopper body keypoints from the current simulator state.

    Default keypoint order:
        0 torso
        1 thigh
        2 leg
        3 foot

    Returns:
        keypoints: array of shape [4, 2] if root_relative=True.
    """
    if body_names is None:
        body_names = DEFAULT_HOPPER_BODIES

    positions = get_body_positions(env, body_names)

    if root_relative:
        return root_relative_keypoints(positions)

    return positions.astype(np.float32)


def get_keypoints(
    env,
    root_relative: bool = True,
) -> np.ndarray:
    """
    Environment-aware keypoint extractor.

    Supported environments:
        Walker2d-v5
        Hopper-v5
    """
    env_id = get_env_id(env)

    if "Walker2d" in env_id:
        return get_walker2d_keypoints(env, root_relative=root_relative)

    if "Hopper" in env_id:
        return get_hopper_keypoints(env, root_relative=root_relative)

    raise ValueError(
        f"Unsupported environment for pose keypoints: {env_id}. "
        "Expected an environment id containing 'Walker2d' or 'Hopper'."
    )


def get_default_body_names(env) -> list[str]:
    """
    Return the default body/keypoint names used by get_keypoints for this env.
    """
    env_id = get_env_id(env)

    if "Walker2d" in env_id:
        return list(DEFAULT_WALKER2D_BODIES)

    if "Hopper" in env_id:
        return list(DEFAULT_HOPPER_BODIES)

    raise ValueError(
        f"Unsupported environment for default body names: {env_id}. "
        "Expected an environment id containing 'Walker2d' or 'Hopper'."
    )