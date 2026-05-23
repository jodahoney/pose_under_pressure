from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.envs.keypoints import get_walker2d_keypoints
from src.rewards.corruptions import add_gaussian_jitter, apply_keypoint_dropout
from src.rewards.pose_reward import pose_reward


class PoseRewardWrapper(gym.Wrapper):
    """
    Replace the environment reward with a pose-derived imitation reward.

    The policy still observes the original simulator observation. Only the
    scalar reward is replaced. Optional corruption can be applied to the
    agent keypoints before reward computation.
    """

    def __init__(
        self,
        env: gym.Env,
        reference_keypoints: np.ndarray,
        alpha: float = 5.0,
        terminate_on_reference_end: bool = True,
        noise_type: str = "clean",
        jitter_sigma: float = 0.0,
        dropout_p: float = 0.0,
        dropout_mode: str = "hold",
        seed: int = 0,
    ):
        super().__init__(env)

        if reference_keypoints.ndim != 3:
            raise ValueError(
                "reference_keypoints must have shape [T, K, D], "
                f"got {reference_keypoints.shape}"
            )

        self.reference_keypoints = reference_keypoints.astype(np.float32)
        self.alpha = alpha
        self.terminate_on_reference_end = terminate_on_reference_end

        self.noise_type = noise_type
        self.jitter_sigma = jitter_sigma
        self.dropout_p = dropout_p
        self.dropout_mode = dropout_mode
        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.previous_clean_keypoints: np.ndarray | None = None

    def reset(self, **kwargs):
        self.t = 0
        self.previous_clean_keypoints = None
        return self.env.reset(**kwargs)

    def _corrupt_keypoints(
        self,
        keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        visibility_mask = np.ones(keypoints.shape[0], dtype=np.float32)

        if self.noise_type == "clean":
            corrupted = keypoints

        elif self.noise_type == "jitter":
            corrupted = add_gaussian_jitter(
                keypoints=keypoints,
                sigma=self.jitter_sigma,
                rng=self.rng,
            )

        elif self.noise_type == "dropout":
            corrupted, visibility_mask = apply_keypoint_dropout(
                keypoints=keypoints,
                p_drop=self.dropout_p,
                rng=self.rng,
                previous_keypoints=self.previous_clean_keypoints,
                mode=self.dropout_mode,
            )

        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        return corrupted.astype(np.float32), visibility_mask.astype(np.float32)

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        clean_agent_keypoints = get_walker2d_keypoints(self.env)
        reward_keypoints, visibility_mask = self._corrupt_keypoints(clean_agent_keypoints)

        ref_t = min(self.t, len(self.reference_keypoints) - 1)
        ref_keypoints = self.reference_keypoints[ref_t]

        imitation_reward = pose_reward(
            agent_keypoints=reward_keypoints,
            reference_keypoints=ref_keypoints,
            alpha=self.alpha,
        )

        clean_keypoint_mse = float(np.mean((clean_agent_keypoints - ref_keypoints) ** 2))
        reward_keypoint_mse = float(np.mean((reward_keypoints - ref_keypoints) ** 2))

        info = dict(info)
        info["env_reward"] = float(env_reward)
        info["pose_reward"] = float(imitation_reward)
        info["clean_keypoint_mse"] = clean_keypoint_mse
        info["reward_keypoint_mse"] = reward_keypoint_mse
        info["keypoint_mse"] = clean_keypoint_mse
        info["reference_t"] = int(ref_t)
        info["noise_type"] = self.noise_type
        info["jitter_sigma"] = float(self.jitter_sigma)
        info["dropout_p"] = float(self.dropout_p)
        info["dropout_mode"] = self.dropout_mode
        info["visibility_frac"] = float(np.mean(visibility_mask))

        self.previous_clean_keypoints = clean_agent_keypoints.copy()
        self.t += 1

        if self.terminate_on_reference_end and self.t >= len(self.reference_keypoints):
            truncated = True

        return obs, imitation_reward, terminated, truncated, info
