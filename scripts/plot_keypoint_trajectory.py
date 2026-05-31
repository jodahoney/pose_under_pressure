import argparse
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.envs.keypoints import get_keypoints


def rollout_policy(
    env_name: str,
    policy_path: Path,
    reference_keypoints: np.ndarray,
    seed: int,
    max_steps: int,
):
    model = PPO.load(policy_path)
    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)

    keypoints = []
    env_rewards = []

    done = False
    t = 0

    while not done and t < max_steps and t < len(reference_keypoints):
        keypoints.append(get_keypoints(env))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        env_rewards.append(float(reward))
        done = terminated or truncated
        t += 1

    env.close()

    return np.asarray(keypoints, dtype=np.float32), np.asarray(env_rewards, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--keypoint-index", type=int, default=-1)
    parser.add_argument(
        "--coord",
        type=int,
        default=1,
        help="Coordinate index after root-relative projection: 0=x, 1=z.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--title", type=str, default="")
    args = parser.parse_args()

    data = np.load(args.reference, allow_pickle=True)
    reference_keypoints = data["keypoints"]

    learned_keypoints, env_rewards = rollout_policy(
        env_name=args.env,
        policy_path=args.policy_path,
        reference_keypoints=reference_keypoints,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    if len(learned_keypoints) == 0:
        raise RuntimeError("Policy rollout produced no keypoints.")

    T = min(len(learned_keypoints), len(reference_keypoints), args.max_steps)
    reference_keypoints = reference_keypoints[:T]
    learned_keypoints = learned_keypoints[:T]

    keypoint_index = args.keypoint_index
    if keypoint_index < 0:
        keypoint_index = reference_keypoints.shape[1] - 1

    if keypoint_index >= reference_keypoints.shape[1]:
        raise ValueError(
            f"keypoint_index={keypoint_index} is invalid for keypoints with "
            f"shape {reference_keypoints.shape}"
        )

    if args.coord >= reference_keypoints.shape[2]:
        raise ValueError(
            f"coord={args.coord} is invalid for keypoints with "
            f"shape {reference_keypoints.shape}"
        )

    coord_name = "x" if args.coord == 0 else "z"

    mse = float(
        np.mean(
            (
                learned_keypoints[:, keypoint_index, args.coord]
                - reference_keypoints[:, keypoint_index, args.coord]
            )
            ** 2
        )
    )

    plt.figure(figsize=(8, 4.5))
    plt.plot(
        np.arange(T),
        reference_keypoints[:, keypoint_index, args.coord],
        label="Expert reference",
        linewidth=2,
    )
    plt.plot(
        np.arange(T),
        learned_keypoints[:, keypoint_index, args.coord],
        label="Learned policy",
        alpha=0.85,
        linewidth=1.75,
    )

    plt.xlabel("Timestep")
    plt.ylabel(f"Root-relative keypoint {keypoint_index} {coord_name}-coordinate")
    plt.title(args.title or f"{args.env}: expert vs learned keypoint trajectory")
    plt.legend()
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    plt.close()

    print(f"Saved {args.out}")
    print(f"Rollout steps plotted: {T}")
    print(f"Trajectory coordinate MSE: {mse:.6f}")
    print(f"Total raw env reward over plotted rollout: {float(np.sum(env_rewards[:T])):.3f}")


if __name__ == "__main__":
    main()
