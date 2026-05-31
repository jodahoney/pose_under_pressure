from __future__ import annotations

import gymnasium as gym
import numpy as np

from src.envs.keypoints import get_walker2d_keypoints
from src.rewards.corruptions import (
    add_gaussian_jitter,
    apply_keypoint_dropout,
    add_systematic_bias,
)
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
        use_masked_reward: bool = False,
        bias_magnitude: float = 0.0,
        bias_axis: int = 0,
        bias_target: str = "none",
        reward_type: str = "squared",
        huber_delta: float = 0.05,
        seed: int = 0,
        debug_bias_indices: bool = True,
    ):
        super().__init__(env)

        if reference_keypoints.ndim != 3:
            raise ValueError(
                "reference_keypoints must have shape [T, K, D], "
                f"got {reference_keypoints.shape}"
            )

        if noise_type not in {"clean", "jitter", "dropout", "bias", "compound"}:
            raise ValueError(f"Unknown noise_type: {noise_type}")

        if dropout_mode not in {"hold", "zero", "mask"}:
            raise ValueError(f"Unknown dropout_mode: {dropout_mode}")

        if bias_target not in {"none", "all", "feet", "legs", "torso"}:
            raise ValueError(f"Unknown bias_target: {bias_target}")

        if reward_type not in {"squared", "huber"}:
            raise ValueError(f"Unknown reward_type: {reward_type}")

        self.reference_keypoints = reference_keypoints.astype(np.float32)
        self.alpha = alpha
        self.terminate_on_reference_end = terminate_on_reference_end

        self.noise_type = noise_type
        self.jitter_sigma = jitter_sigma
        self.dropout_p = dropout_p
        self.dropout_mode = dropout_mode
        self.use_masked_reward = use_masked_reward

        self.bias_magnitude = bias_magnitude
        self.bias_axis = bias_axis
        self.bias_target = bias_target

        self.reward_type = reward_type
        self.huber_delta = huber_delta

        self.debug_bias_indices = debug_bias_indices
        self._printed_bias_debug = False

        self.rng = np.random.default_rng(seed)

        self.t = 0
        self.previous_clean_keypoints: np.ndarray | None = None

    def reset(self, **kwargs):
        self.t = 0
        self.previous_clean_keypoints = None
        return self.env.reset(**kwargs)

    def _get_keypoint_names(self) -> list[str]:
        """
        Best-effort helper for selecting biased keypoints by MuJoCo body name.

        This may not perfectly align with get_walker2d_keypoints if the keypoint
        extractor uses a custom subset/order. The fallback index rules below
        keep the experiment runnable even if names do not align.
        """
        model = self.env.unwrapped.model
        names = []

        for i in range(model.nbody):
            try:
                name = model.body(i).name
            except Exception:
                name = ""
            names.append(name)

        return names

    def _select_bias_indices(self, keypoints: np.ndarray) -> list[int]:
        """
        Select which keypoints receive systematic bias.

        Walker2d keypoint order from get_walker2d_keypoints:
            0 torso
            1 thigh
            2 leg
            3 foot
            4 thigh_left
            5 leg_left
            6 foot_left
        """
        num_keypoints = keypoints.shape[0]

        if self.bias_target == "none" or self.bias_magnitude == 0.0:
            return []

        if self.bias_target == "all":
            return list(range(num_keypoints))

        if num_keypoints == 7:
            if self.bias_target == "feet":
                return [3, 6]

            if self.bias_target == "legs":
                return [1, 2, 3, 4, 5, 6]

            if self.bias_target == "torso":
                return [0]

        # Generic fallbacks for non-Walker2d keypoint arrays.
        if self.bias_target == "feet":
            return list(range(max(0, num_keypoints - 2), num_keypoints))

        if self.bias_target == "legs":
            return list(range(num_keypoints // 2, num_keypoints))

        if self.bias_target == "torso":
            return list(range(min(1, num_keypoints)))

        return []

    def _maybe_print_bias_debug(self, keypoints: np.ndarray, bias_indices: list[int]) -> None:
        if not self.debug_bias_indices:
            return

        if self._printed_bias_debug:
            return

        if self.bias_target == "none" or self.bias_magnitude == 0.0:
            return

        walker2d_names = [
            "torso",
            "thigh",
            "leg",
            "foot",
            "thigh_left",
            "leg_left",
            "foot_left",
        ]

        selected_names = []
        for idx in bias_indices:
            if keypoints.shape[0] == 7 and 0 <= idx < len(walker2d_names):
                selected_names.append(walker2d_names[idx])
            else:
                selected_names.append(f"keypoint_{idx}")

        print(
            "[PoseRewardWrapper] "
            f"bias_target={self.bias_target}, "
            f"bias_magnitude={self.bias_magnitude}, "
            f"bias_axis={self.bias_axis}, "
            f"num_keypoints={keypoints.shape[0]}, "
            f"bias_indices={bias_indices}, "
            f"selected_names={selected_names}"
        )

        self._printed_bias_debug = True

    def _apply_bias(self, keypoints: np.ndarray) -> np.ndarray:
        corrupted = keypoints.copy()

        if self.bias_magnitude == 0.0 or self.bias_target == "none":
            return corrupted

        if self.bias_axis < 0 or self.bias_axis >= corrupted.shape[1]:
            raise ValueError(
                f"bias_axis={self.bias_axis} is invalid for keypoints with "
                f"dimension {corrupted.shape[1]}"
            )

        bias_indices = self._select_bias_indices(corrupted)
        self._maybe_print_bias_debug(corrupted, bias_indices)

        if len(bias_indices) > 0:
            corrupted = add_systematic_bias(
                keypoints=corrupted,
                bias_indices=bias_indices,
                bias_magnitude=self.bias_magnitude,
                bias_axis=self.bias_axis,
            )

        return corrupted.astype(np.float32)

    def _compute_pose_reward(
        self,
        reward_keypoints: np.ndarray,
        ref_keypoints: np.ndarray,
        visibility_mask: np.ndarray,
    ) -> float:
        """
        Compute squared or Huber pose reward, optionally using a visibility mask.

        The standard unmasked squared case calls the original pose_reward
        function to preserve previous behavior.
        """
        mask = visibility_mask if self.use_masked_reward else None

        if self.reward_type == "squared" and mask is None:
            return float(
                pose_reward(
                    agent_keypoints=reward_keypoints,
                    reference_keypoints=ref_keypoints,
                    alpha=self.alpha,
                )
            )

        diff = reward_keypoints - ref_keypoints

        if self.reward_type == "squared":
            per_keypoint_loss = np.sum(diff ** 2, axis=-1)

        elif self.reward_type == "huber":
            abs_diff = np.abs(diff)
            quadratic = np.minimum(abs_diff, self.huber_delta)
            linear = abs_diff - quadratic
            huber = 0.5 * quadratic ** 2 + self.huber_delta * linear
            per_keypoint_loss = np.sum(huber, axis=-1)

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        if mask is not None:
            denom = float(np.sum(mask)) + 1e-8
            loss = float(np.sum(per_keypoint_loss * mask) / denom)
        else:
            loss = float(np.sum(per_keypoint_loss))

        return float(np.exp(-self.alpha * loss))

    def _corrupt_keypoints(
        self,
        keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        visibility_mask = np.ones(keypoints.shape[0], dtype=np.float32)
        corrupted = keypoints.copy()

        if self.noise_type == "clean":
            pass

        elif self.noise_type == "jitter":
            corrupted = add_gaussian_jitter(
                keypoints=corrupted,
                sigma=self.jitter_sigma,
                rng=self.rng,
            )

        elif self.noise_type == "dropout":
            corrupted, visibility_mask = apply_keypoint_dropout(
                keypoints=corrupted,
                p_drop=self.dropout_p,
                rng=self.rng,
                previous_keypoints=self.previous_clean_keypoints,
                mode=self.dropout_mode,
            )

        elif self.noise_type == "bias":
            corrupted = self._apply_bias(corrupted)

        elif self.noise_type == "compound":
            corrupted = add_gaussian_jitter(
                keypoints=corrupted,
                sigma=self.jitter_sigma,
                rng=self.rng,
            )

            corrupted = self._apply_bias(corrupted)

            corrupted, visibility_mask = apply_keypoint_dropout(
                keypoints=corrupted,
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

        imitation_reward = self._compute_pose_reward(
            reward_keypoints=reward_keypoints,
            ref_keypoints=ref_keypoints,
            visibility_mask=visibility_mask,
        )

        clean_keypoint_mse = float(np.mean((clean_agent_keypoints - ref_keypoints) ** 2))
        reward_keypoint_mse = float(np.mean((reward_keypoints - ref_keypoints) ** 2))

        if self.use_masked_reward:
            masked_reward_keypoint_mse = float(
                np.sum(
                    np.mean((reward_keypoints - ref_keypoints) ** 2, axis=-1)
                    * visibility_mask
                )
                / (np.sum(visibility_mask) + 1e-8)
            )
        else:
            masked_reward_keypoint_mse = reward_keypoint_mse

        info = dict(info)
        info["env_reward"] = float(env_reward)
        info["pose_reward"] = float(imitation_reward)
        info["clean_keypoint_mse"] = clean_keypoint_mse
        info["reward_keypoint_mse"] = reward_keypoint_mse
        info["masked_reward_keypoint_mse"] = masked_reward_keypoint_mse
        info["keypoint_mse"] = clean_keypoint_mse
        info["reference_t"] = int(ref_t)

        info["noise_type"] = self.noise_type
        info["jitter_sigma"] = float(self.jitter_sigma)
        info["dropout_p"] = float(self.dropout_p)
        info["dropout_mode"] = self.dropout_mode
        info["use_masked_reward"] = bool(self.use_masked_reward)
        info["bias_magnitude"] = float(self.bias_magnitude)
        info["bias_axis"] = int(self.bias_axis)
        info["bias_target"] = self.bias_target
        info["reward_type"] = self.reward_type
        info["huber_delta"] = float(self.huber_delta)

        info["visibility_frac"] = float(np.mean(visibility_mask))
        info["mask_mean"] = float(np.mean(visibility_mask))

        self.previous_clean_keypoints = clean_agent_keypoints.copy()
        self.t += 1

        if self.terminate_on_reference_end and self.t >= len(self.reference_keypoints):
            truncated = True

        return obs, imitation_reward, terminated, truncated, info