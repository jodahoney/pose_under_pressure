import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.envs.wrappers import PoseRewardWrapper


def make_pose_env(
    env_name: str,
    reference_path: Path,
    seed: int,
    alpha: float,
    terminate_on_reference_end: bool,
    noise_type: str,
    jitter_sigma: float,
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
    }


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
    parser.add_argument("--noise-type", type=str, default="clean", choices=["clean", "jitter"])
    parser.add_argument("--jitter-sigma", type=float, default=0.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([
        make_pose_env(
            env_name=args.env,
            reference_path=args.reference,
            seed=args.seed,
            alpha=args.alpha,
            terminate_on_reference_end=True,
            noise_type=args.noise_type,
            jitter_sigma=args.jitter_sigma,
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

    metrics = evaluate_model(
        model=model,
        env_name=args.env,
        reference_path=args.reference,
        seed=args.seed + 10_000,
        alpha=args.alpha,
        episodes=args.eval_episodes,
        noise_type="clean",
        jitter_sigma=0.0,
    )

    metrics.update({
        "env": args.env,
        "reference": str(args.reference),
        "seed": args.seed,
        "total_steps": args.total_steps,
        "alpha": args.alpha,
        "noise_type": args.noise_type,
        "jitter_sigma": args.jitter_sigma,
        "policy_path": str(model_path),
    })

    metrics_path = args.out_dir / "eval_metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    print(f"Saved pose-reward PPO policy to {model_path}")
    print(f"Saved eval metrics to {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
