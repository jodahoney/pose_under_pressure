import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.envs.keypoints import get_default_body_names, get_keypoints


def load_policy(policy_path: Path | None):
    if policy_path is None:
        return None

    from stable_baselines3 import PPO

    return PPO.load(policy_path)


def collect_reference(
    env_name: str,
    out_path: Path,
    seed: int,
    max_steps: int,
    policy_path: Path | None = None,
    deterministic: bool = True,
):
    env = gym.make(env_name)
    obs, info = env.reset(seed=seed)

    policy = load_policy(policy_path)

    obs_list = []
    action_list = []
    reward_list = []
    keypoint_list = []

    body_names = get_default_body_names(env)

    for _ in range(max_steps):
        keypoints = get_keypoints(env)

        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)

        obs_list.append(obs)
        action_list.append(action)
        keypoint_list.append(keypoints)

        obs, reward, terminated, truncated, info = env.step(action)
        reward_list.append(reward)

        if terminated or truncated:
            break

    env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_path,
        obs=np.asarray(obs_list, dtype=np.float32),
        actions=np.asarray(action_list, dtype=np.float32),
        env_rewards=np.asarray(reward_list, dtype=np.float32),
        keypoints=np.asarray(keypoint_list, dtype=np.float32),
        body_names=np.asarray(body_names),
        env_name=env_name,
        seed=seed,
        policy_path=str(policy_path) if policy_path is not None else "",
        deterministic=deterministic,
    )

    print(f"Saved reference to {out_path}")
    print(f"Environment: {env_name}")
    print(f"Steps: {len(obs_list)}")
    print(f"Total env reward: {np.sum(reward_list):.3f}")
    print(f"Mean env reward: {np.mean(reward_list):.3f}")
    print(f"Body names: {body_names}")
    print(f"Keypoints shape: {np.asarray(keypoint_list).shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Walker2d-v5")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/references/walker2d_v5_reference_random_seed0.npz"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    collect_reference(
        env_name=args.env,
        out_path=args.out,
        seed=args.seed,
        max_steps=args.max_steps,
        policy_path=args.policy_path,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()