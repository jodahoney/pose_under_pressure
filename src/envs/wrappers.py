from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.envs.keypoints import get_walker2d_keypoints
from src.rewards.corruptions import add_gaussian_jitter
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
        self.rng = np.random.default_rng(seed)

        self.t = 0

    def reset(self, **kwargs):
        self.t = 0
        return self.env.reset(**kwargs)

    def _corrupt_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        if self.noise_type == "clean":
            return keypoints

        if self.noise_type == "jitter":
            return add_gaussian_jitter(
                keypoints=keypoints,
                sigma=self.jitter_sigma,
                rng=self.rng,
            )

        raise ValueError(f"Unknown noise_type: {self.noise_type}")

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        clean_agent_keypoints = get_walker2d_keypoints(self.env)
        reward_keypoints = self._corrupt_keypoints(clean_agent_keypoints)

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

        self.t += 1

        if self.terminate_on_reference_end and self.t >= len(self.reference_keypoints):
            truncated = True

        return obs, imitation_reward, terminated, truncated, info
