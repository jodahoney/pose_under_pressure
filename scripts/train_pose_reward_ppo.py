import argparse
import csv
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.envs.wrappers import PoseRewardWrapper


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
    "jitter_huber_sigma_0.10": {
        "noise_type": "jitter",
        "jitter_sigma": 0.10,
        "dropout_p": 0.0,
        "dropout_mode": "hold",
        "use_masked_reward": False,
        "bias_magnitude": 0.0,
        "bias_axis": 0,
        "bias_target": "none",
        "reward_type": "huber",
    },
}


def apply_condition_preset(args):
    """
    Fill corruption/reward arguments from --condition unless the user explicitly
    provided manual low-level arguments.
    """
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


def make_pose_env(
    env_name: str,
    reference_path: Path,
    seed: int,
    alpha: float,
    terminate_on_reference_end: bool,
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
    def _init():
        data = np.load(reference_path, allow_pickle=True)
        reference_keypoints = data["keypoints"]

        env = gym.make(env_name)
        env = PoseRewardWrapper(
            env,
            reference_keypoints=reference_keypoints,
            alpha=alpha,
            terminate_on_reference_end=terminate_on_reference_end,
            noise_type=noise_type,
            jitter_sigma=jitter_sigma,
            dropout_p=dropout_p,
            dropout_mode=dropout_mode,
            use_masked_reward=use_masked_reward,
            bias_magnitude=bias_magnitude,
            bias_axis=bias_axis,
            bias_target=bias_target,
            reward_type=reward_type,
            huber_delta=huber_delta,
            seed=seed,
        )
        env.reset(seed=seed)
        return env

    return _init


def evaluate_model(
    model: PPO,
    env_name: str,
    reference_path: Path,
    seed: int,
    alpha: float,
    episodes: int,
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
):
    data = np.load(reference_path, allow_pickle=True)
    reference_keypoints = data["keypoints"]

    returns = []
    env_returns = []
    keypoint_mses = []
    lengths = []

    for ep in range(episodes):
        env = gym.make(env_name)
        env = PoseRewardWrapper(
            env,
            reference_keypoints=reference_keypoints,
            alpha=alpha,
            terminate_on_reference_end=True,
            noise_type=noise_type,
            jitter_sigma=jitter_sigma,
            dropout_p=dropout_p,
            dropout_mode=dropout_mode,
            use_masked_reward=use_masked_reward,
            bias_magnitude=bias_magnitude,
            bias_axis=bias_axis,
            bias_target=bias_target,
            reward_type=reward_type,
            huber_delta=huber_delta,
            seed=seed + ep,
        )

        obs, info = env.reset(seed=seed + ep)
        done = False

        ep_pose_return = 0.0
        ep_env_return = 0.0
        ep_keypoint_mses = []
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_pose_return += float(reward)
            ep_env_return += float(info["env_reward"])
            ep_keypoint_mses.append(float(info["keypoint_mse"]))
            ep_len += 1

            done = terminated or truncated

        env.close()

        returns.append(ep_pose_return)
        env_returns.append(ep_env_return)
        keypoint_mses.append(float(np.mean(ep_keypoint_mses)))
        lengths.append(ep_len)

    return {
        "eval_pose_return_mean": float(np.mean(returns)),
        "eval_pose_return_std": float(np.std(returns)),
        "eval_env_return_mean": float(np.mean(env_returns)),
        "eval_env_return_std": float(np.std(env_returns)),
        "eval_keypoint_mse_mean": float(np.mean(keypoint_mses)),
        "eval_keypoint_mse_std": float(np.std(keypoint_mses)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_length_std": float(np.std(lengths)),
    }


def reward_sanity_check(
    env_name: str,
    reference_path: Path,
    seed: int,
    alpha: float,
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
    num_steps: int = 200,
):
    """
    Runs a short random-policy rollout and prints reward diagnostics.
    This helps catch all-zero/all-constant rewards before a long PPO run.
    """
    data = np.load(reference_path, allow_pickle=True)
    reference_keypoints = data["keypoints"]

    env = gym.make(env_name)
    env = PoseRewardWrapper(
        env,
        reference_keypoints=reference_keypoints,
        alpha=alpha,
        terminate_on_reference_end=True,
        noise_type=noise_type,
        jitter_sigma=jitter_sigma,
        dropout_p=dropout_p,
        dropout_mode=dropout_mode,
        use_masked_reward=use_masked_reward,
        bias_magnitude=bias_magnitude,
        bias_axis=bias_axis,
        bias_target=bias_target,
        reward_type=reward_type,
        huber_delta=huber_delta,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)
    rewards = []
    mses = []
    mask_means = []

    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))

        if "keypoint_mse" in info:
            mses.append(float(info["keypoint_mse"]))

        if "mask_mean" in info:
            mask_means.append(float(info["mask_mean"]))

        if terminated or truncated:
            obs, info = env.reset(seed=seed + 1)

    env.close()

    rewards = np.asarray(rewards, dtype=np.float32)

    print("\nReward sanity check")
    print("-------------------")
    print(f"noise_type: {noise_type}")
    print(f"jitter_sigma: {jitter_sigma}")
    print(f"dropout_p: {dropout_p}")
    print(f"dropout_mode: {dropout_mode}")
    print(f"use_masked_reward: {use_masked_reward}")
    print(f"bias_magnitude: {bias_magnitude}")
    print(f"bias_axis: {bias_axis}")
    print(f"bias_target: {bias_target}")
    print(f"reward_type: {reward_type}")
    print(f"reward mean: {float(np.mean(rewards)):.6f}")
    print(f"reward std:  {float(np.std(rewards)):.6f}")
    print(f"reward min:  {float(np.min(rewards)):.6f}")
    print(f"reward max:  {float(np.max(rewards)):.6f}")

    if mses:
        print(f"keypoint mse mean: {float(np.mean(mses)):.6f}")

    if mask_means:
        print(f"mask mean: {float(np.mean(mask_means)):.6f}")

    print()


def write_metadata(args, model_path: Path, metrics_path: Path):
    metadata = {
        "condition": args.condition,
        "env": args.env,
        "reference": str(args.reference),
        "seed": args.seed,
        "total_steps": args.total_steps,
        "alpha": args.alpha,
        "noise_type": args.noise_type,
        "jitter_sigma": args.jitter_sigma,
        "dropout_p": args.dropout_p,
        "dropout_mode": args.dropout_mode,
        "use_masked_reward": args.use_masked_reward,
        "bias_magnitude": args.bias_magnitude,
        "bias_axis": args.bias_axis,
        "bias_target": args.bias_target,
        "reward_type": args.reward_type,
        "huber_delta": args.huber_delta,
        "policy_path": str(model_path),
        "metrics_path": str(metrics_path),
    }

    metadata_path = args.out_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="Walker2d-v5")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Named condition preset. If provided, overrides the low-level corruption args.",
    )

    parser.add_argument(
        "--noise-type",
        type=str,
        default="clean",
        choices=["clean", "jitter", "dropout", "bias", "compound"],
    )
    parser.add_argument("--jitter-sigma", type=float, default=0.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--dropout-mode", type=str, default="hold", choices=["hold", "zero", "mask"])

    parser.add_argument(
        "--use-masked-reward",
        action="store_true",
        help="If set, keypoints dropped with dropout_mode=mask are ignored in the reward.",
    )

    parser.add_argument("--bias-magnitude", type=float, default=0.0)
    parser.add_argument("--bias-axis", type=int, default=0)
    parser.add_argument(
        "--bias-target",
        type=str,
        default="none",
        choices=["none", "all", "feet", "legs", "torso"],
    )

    parser.add_argument("--reward-type", type=str, default="squared", choices=["squared", "huber"])
    parser.add_argument("--huber-delta", type=float, default=0.05)

    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip the short random-policy reward sanity check.",
    )

    args = parser.parse_args()
    args = apply_condition_preset(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_sanity_check:
        reward_sanity_check(
            env_name=args.env,
            reference_path=args.reference,
            seed=args.seed,
            alpha=args.alpha,
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

    env = DummyVecEnv([
        make_pose_env(
            env_name=args.env,
            reference_path=args.reference,
            seed=args.seed,
            alpha=args.alpha,
            terminate_on_reference_end=True,
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
    ])
    env = VecMonitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(args.out_dir / "tb"),
        device=args.device,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=args.total_steps)

    model_path = args.out_dir / "policy.zip"
    model.save(model_path)
    env.close()

    # Final evaluation is intentionally always clean.
    # This measures whether a policy trained under corrupted rewards imitates the
    # clean reference trajectory under ground-truth simulator keypoints.
    metrics = evaluate_model(
        model=model,
        env_name=args.env,
        reference_path=args.reference,
        seed=args.seed + 10_000,
        alpha=args.alpha,
        episodes=args.eval_episodes,
        noise_type="clean",
        jitter_sigma=0.0,
        dropout_p=0.0,
        dropout_mode="hold",
        use_masked_reward=False,
        bias_magnitude=0.0,
        bias_axis=0,
        bias_target="none",
        reward_type="squared",
        huber_delta=args.huber_delta,
    )

    metrics.update({
        "condition": args.condition if args.condition is not None else "manual",
        "env": args.env,
        "reference": str(args.reference),
        "seed": args.seed,
        "total_steps": args.total_steps,
        "alpha": args.alpha,
        "noise_type": args.noise_type,
        "jitter_sigma": args.jitter_sigma,
        "dropout_p": args.dropout_p,
        "dropout_mode": args.dropout_mode,
        "use_masked_reward": args.use_masked_reward,
        "bias_magnitude": args.bias_magnitude,
        "bias_axis": args.bias_axis,
        "bias_target": args.bias_target,
        "reward_type": args.reward_type,
        "huber_delta": args.huber_delta,
        "policy_path": str(model_path),
    })

    metrics_path = args.out_dir / "eval_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    metadata_path = write_metadata(args, model_path=model_path, metrics_path=metrics_path)

    print(f"Saved pose-reward PPO policy to {model_path}")
    print(f"Saved eval metrics to {metrics_path}")
    print(f"Saved metadata to {metadata_path}")
    print(metrics)


if __name__ == "__main__":
    main()