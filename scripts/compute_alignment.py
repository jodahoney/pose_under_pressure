import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.envs.keypoints import get_walker2d_keypoints
from src.rewards.corruptions import (
    add_gaussian_jitter,
    apply_keypoint_dropout,
    add_systematic_bias,
)
from src.rewards.pose_reward import pose_reward


CONDITION_PRESETS = {
    "clean": {
        "noise_type": "clean",
        "jitter_sigma": 0.0,
        "dropout_p": 0.0,
        "dropout_mode": "hold",
        "use_masked_reward": False,
        "bias_magnitude": 0.0,
        "bias_axis": 0,
        "bias_target": "none",
        "reward_type": "squared",
    },
    "jitter_sigma_0.05": {
        "noise_type": "jitter",
        "jitter_sigma": 0.05,
        "dropout_p": 0.0,
        "dropout_mode": "hold",
        "use_masked_reward": False,
        "bias_magnitude": 0.0,
        "bias_axis": 0,
        "bias_target": "none",
        "reward_type": "squared",
    },
    "jitter_sigma_0.10": {
        "noise_type": "jitter",
        "jitter_sigma": 0.10,
        "dropout_p": 0.0,
        "dropout_mode": "hold",
        "use_masked_reward": False,
        "bias_magnitude": 0.0,
        "bias_axis": 0,
        "bias_target": "none",
        "reward_type": "squared",
    },
    "dropout_hold_p_0.10": {
        "noise_type": "dropout",
        "jitter_sigma": 0.0,
        "dropout_p": 0.10,
        "dropout_mode": "hold",
        "use_masked_reward": False,
        "bias_magnitude": 0.0,
        "bias_axis": 0,
        "bias_target": "none",
        "reward_type": "squared",
    },
    "dropout_hold_p_0.30": {
        "noise_type": "dropout",
        "jitter_sigma": 0.0,
        "dropout_p": 0.30,
        "dropout_mode": "hold",
        "use_masked_reward": False,
        "bias_magnitude": 0.0,
        "bias_axis": 0,
        "bias_target": "none",
        "reward_type": "squared",
    },
    "dropout_zero_p_0.30": {
        "noise_type": "dropout",
        "jitter_sigma": 0.0,
        "dropout_p": 0.30,
        "dropout_mode": "zero",
        "use_masked_reward": False,
        "bias_magnitude": 0.0,
        "bias_axis": 0,
        "bias_target": "none",
        "reward_type": "squared",
    },
    "dropout_masked_p_0.30": {
        "noise_type": "dropout",
        "jitter_sigma": 0.0,
        "dropout_p": 0.30,
        "dropout_mode": "mask",
        "use_masked_reward": True,
        "bias_magnitude": 0.0,
        "bias_axis": 0,
        "bias_target": "none",
        "reward_type": "squared",
    },
    "bias_feet_0.05": {
        "noise_type": "bias",
        "jitter_sigma": 0.0,
        "dropout_p": 0.0,
        "dropout_mode": "hold",
        "use_masked_reward": False,
        "bias_magnitude": 0.05,
        "bias_axis": 0,
        "bias_target": "feet",
        "reward_type": "squared",
    },
    "bias_feet_0.10": {
        "noise_type": "bias",
        "jitter_sigma": 0.0,
        "dropout_p": 0.0,
        "dropout_mode": "hold",
        "use_masked_reward": False,
        "bias_magnitude": 0.10,
        "bias_axis": 0,
        "bias_target": "feet",
        "reward_type": "squared",
    },
}


def apply_condition_preset(args):
    if args.condition is None:
        return args

    if args.condition not in CONDITION_PRESETS:
        valid = ", ".join(sorted(CONDITION_PRESETS.keys()))
        raise ValueError(f"Unknown condition '{args.condition}'. Valid conditions: {valid}")

    preset = CONDITION_PRESETS[args.condition]

    args.noise_type = preset["noise_type"]
    args.jitter_sigma = preset["jitter_sigma"]
    args.dropout_p = preset["dropout_p"]
    args.dropout_mode = preset["dropout_mode"]
    args.use_masked_reward = preset["use_masked_reward"]
    args.bias_magnitude = preset["bias_magnitude"]
    args.bias_axis = preset["bias_axis"]
    args.bias_target = preset["bias_target"]
    args.reward_type = preset["reward_type"]

    return args


def select_bias_indices(
    keypoints: np.ndarray,
    bias_target: str,
    bias_magnitude: float,
) -> list[int]:
    """
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

    if bias_target == "none" or bias_magnitude == 0.0:
        return []

    if bias_target == "all":
        return list(range(num_keypoints))

    if num_keypoints == 7:
        if bias_target == "feet":
            return [3, 6]
        if bias_target == "legs":
            return [1, 2, 3, 4, 5, 6]
        if bias_target == "torso":
            return [0]

    if bias_target == "feet":
        return list(range(max(0, num_keypoints - 2), num_keypoints))
    if bias_target == "legs":
        return list(range(num_keypoints // 2, num_keypoints))
    if bias_target == "torso":
        return list(range(min(1, num_keypoints)))

    return []


def compute_pose_reward(
    agent_keypoints: np.ndarray,
    reference_keypoints: np.ndarray,
    alpha: float,
    visibility_mask: np.ndarray | None = None,
    use_masked_reward: bool = False,
    reward_type: str = "squared",
    huber_delta: float = 0.05,
) -> float:
    """
    Match PoseRewardWrapper reward behavior for clean/noisy alignment.
    """
    mask = visibility_mask if use_masked_reward else None

    if reward_type == "squared" and mask is None:
        return float(
            pose_reward(
                agent_keypoints=agent_keypoints,
                reference_keypoints=reference_keypoints,
                alpha=alpha,
            )
        )

    diff = agent_keypoints - reference_keypoints

    if reward_type == "squared":
        per_keypoint_loss = np.sum(diff ** 2, axis=-1)

    elif reward_type == "huber":
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, huber_delta)
        linear = abs_diff - quadratic
        huber = 0.5 * quadratic ** 2 + huber_delta * linear
        per_keypoint_loss = np.sum(huber, axis=-1)

    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    if mask is not None:
        loss = float(np.sum(per_keypoint_loss * mask) / (np.sum(mask) + 1e-8))
    else:
        loss = float(np.sum(per_keypoint_loss))

    return float(np.exp(-alpha * loss))


def corrupt_keypoints(
    keypoints: np.ndarray,
    previous_keypoints: np.ndarray | None,
    noise_type: str,
    jitter_sigma: float,
    dropout_p: float,
    dropout_mode: str,
    bias_magnitude: float,
    bias_axis: int,
    bias_target: str,
    rng: np.random.Generator,
):
    visibility_mask = np.ones(keypoints.shape[0], dtype=np.float32)
    corrupted = keypoints.copy().astype(np.float32)

    if noise_type == "clean":
        pass

    elif noise_type == "jitter":
        corrupted = add_gaussian_jitter(corrupted, jitter_sigma, rng)

    elif noise_type == "dropout":
        corrupted, visibility_mask = apply_keypoint_dropout(
            keypoints=corrupted,
            p_drop=dropout_p,
            rng=rng,
            previous_keypoints=previous_keypoints,
            mode=dropout_mode,
        )

    elif noise_type == "bias":
        bias_indices = select_bias_indices(
            keypoints=corrupted,
            bias_target=bias_target,
            bias_magnitude=bias_magnitude,
        )
        corrupted = add_systematic_bias(
            keypoints=corrupted,
            bias_indices=bias_indices,
            bias_magnitude=bias_magnitude,
            bias_axis=bias_axis,
        )

    elif noise_type == "compound":
        corrupted = add_gaussian_jitter(corrupted, jitter_sigma, rng)

        bias_indices = select_bias_indices(
            keypoints=corrupted,
            bias_target=bias_target,
            bias_magnitude=bias_magnitude,
        )
        corrupted = add_systematic_bias(
            keypoints=corrupted,
            bias_indices=bias_indices,
            bias_magnitude=bias_magnitude,
            bias_axis=bias_axis,
        )

        corrupted, visibility_mask = apply_keypoint_dropout(
            keypoints=corrupted,
            p_drop=dropout_p,
            rng=rng,
            previous_keypoints=previous_keypoints,
            mode=dropout_mode,
        )

    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    return corrupted.astype(np.float32), visibility_mask.astype(np.float32)


def compute_alignment(
    env_name: str,
    reference_path: Path,
    policy_path: Path,
    out_path: Path,
    seed: int,
    episodes: int,
    alpha: float,
    condition: str | None,
    noise_type: str,
    jitter_sigma: float,
    dropout_p: float,
    dropout_mode: str,
    use_masked_reward: bool,
    bias_magnitude: float,
    bias_axis: int,
    bias_target: str,
    reward_type: str,
    huber_delta: float,
):
    data = np.load(reference_path, allow_pickle=True)
    reference_keypoints = data["keypoints"]

    model = PPO.load(policy_path)
    rng = np.random.default_rng(seed)

    rows = []

    for ep in range(episodes):
        env = gym.make(env_name)
        obs, info = env.reset(seed=seed + ep)

        done = False
        t = 0
        previous_keypoints = None

        while not done and t < len(reference_keypoints):
            action, _ = model.predict(obs, deterministic=True)
            obs, env_reward, terminated, truncated, info = env.step(action)

            clean_keypoints = get_walker2d_keypoints(env)
            ref_keypoints = reference_keypoints[t]

            noisy_keypoints, visibility_mask = corrupt_keypoints(
                keypoints=clean_keypoints,
                previous_keypoints=previous_keypoints,
                noise_type=noise_type,
                jitter_sigma=jitter_sigma,
                dropout_p=dropout_p,
                dropout_mode=dropout_mode,
                bias_magnitude=bias_magnitude,
                bias_axis=bias_axis,
                bias_target=bias_target,
                rng=rng,
            )

            clean_reward = compute_pose_reward(
                agent_keypoints=clean_keypoints,
                reference_keypoints=ref_keypoints,
                alpha=alpha,
                visibility_mask=None,
                use_masked_reward=False,
                reward_type="squared",
                huber_delta=huber_delta,
            )

            noisy_reward = compute_pose_reward(
                agent_keypoints=noisy_keypoints,
                reference_keypoints=ref_keypoints,
                alpha=alpha,
                visibility_mask=visibility_mask,
                use_masked_reward=use_masked_reward,
                reward_type=reward_type,
                huber_delta=huber_delta,
            )

            rows.append({
                "episode": ep,
                "t": t,
                "clean_reward": clean_reward,
                "noisy_reward": noisy_reward,
                "env_reward": float(env_reward),
                "clean_keypoint_mse": float(np.mean((clean_keypoints - ref_keypoints) ** 2)),
                "noisy_keypoint_mse": float(np.mean((noisy_keypoints - ref_keypoints) ** 2)),
                "masked_noisy_keypoint_mse": float(
                    np.sum(
                        np.mean((noisy_keypoints - ref_keypoints) ** 2, axis=-1)
                        * visibility_mask
                    )
                    / (np.sum(visibility_mask) + 1e-8)
                ),
                "visibility_frac": float(np.mean(visibility_mask)),
            })

            previous_keypoints = clean_keypoints.copy()
            t += 1
            done = terminated or truncated

        env.close()

    df = pd.DataFrame(rows)

    if len(df) < 2:
        raise RuntimeError("Not enough samples to compute alignment.")

    pearson = pearsonr(df["clean_reward"], df["noisy_reward"])
    spearman = spearmanr(df["clean_reward"], df["noisy_reward"])

    summary = pd.DataFrame([{
        "condition": condition if condition is not None else "manual",
        "env": env_name,
        "reference": str(reference_path),
        "policy_path": str(policy_path),
        "seed": seed,
        "episodes": episodes,
        "alpha": alpha,
        "noise_type": noise_type,
        "jitter_sigma": jitter_sigma,
        "dropout_p": dropout_p,
        "dropout_mode": dropout_mode,
        "use_masked_reward": use_masked_reward,
        "bias_magnitude": bias_magnitude,
        "bias_axis": bias_axis,
        "bias_target": bias_target,
        "reward_type": reward_type,
        "huber_delta": huber_delta,
        "n_samples": len(df),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "mean_clean_reward": float(df["clean_reward"].mean()),
        "mean_noisy_reward": float(df["noisy_reward"].mean()),
        "mean_clean_keypoint_mse": float(df["clean_keypoint_mse"].mean()),
        "mean_noisy_keypoint_mse": float(df["noisy_keypoint_mse"].mean()),
        "mean_masked_noisy_keypoint_mse": float(df["masked_noisy_keypoint_mse"].mean()),
        "mean_visibility_frac": float(df["visibility_frac"].mean()),
    }])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples_path = out_path.with_name(out_path.stem + "_samples.csv")
    df.to_csv(samples_path, index=False)
    summary.to_csv(out_path, index=False)

    print(f"Wrote alignment summary: {out_path}")
    print(f"Wrote alignment samples: {samples_path}")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Walker2d-v5")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=5.0)

    parser.add_argument("--condition", type=str, default=None)

    parser.add_argument(
        "--noise-type",
        type=str,
        default="jitter",
        choices=["clean", "jitter", "dropout", "bias", "compound"],
    )
    parser.add_argument("--jitter-sigma", type=float, default=0.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--dropout-mode", type=str, default="hold", choices=["hold", "zero", "mask"])
    parser.add_argument("--use-masked-reward", action="store_true")

    parser.add_argument("--bias-magnitude", type=float, default=0.0)
    parser.add_argument("--bias-axis", type=int, default=0)
    parser.add_argument("--bias-target", type=str, default="none", choices=["none", "all", "feet", "legs", "torso"])

    parser.add_argument("--reward-type", type=str, default="squared", choices=["squared", "huber"])
    parser.add_argument("--huber-delta", type=float, default=0.05)

    args = parser.parse_args()
    args = apply_condition_preset(args)

    compute_alignment(
        env_name=args.env,
        reference_path=args.reference,
        policy_path=args.policy_path,
        out_path=args.out,
        seed=args.seed,
        episodes=args.episodes,
        alpha=args.alpha,
        condition=args.condition,
        noise_type=args.noise_type,
        jitter_sigma=args.jitter_sigma,
        dropout_p=args.dropout_p,
        dropout_mode=args.dropout_mode,
        use_masked_reward=args.use_masked_reward,
        bias_magnitude=args.bias_magnitude,
        bias_axis=args.bias_axis,
        bias_target=args.bias_target,
        reward_type=args.reward_type,
        huber_delta=args.huber_delta,
    )


if __name__ == "__main__":
    main()