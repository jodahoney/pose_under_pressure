import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def evaluate_policy(
    env_name: str,
    policy_path: Path,
    seed: int,
    episodes: int,
    deterministic: bool,
):
    model = PPO.load(policy_path)

    returns = []
    lengths = []

    for ep in range(episodes):
        env = gym.make(env_name)
        obs, info = env.reset(seed=seed + ep)

        done = False
        ep_return = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)

            ep_return += reward
            ep_len += 1
            done = terminated or truncated

        env.close()

        returns.append(ep_return)
        lengths.append(ep_len)

        print(f"episode={ep}, return={ep_return:.3f}, length={ep_len}")

    print("\nSummary")
    print(f"Mean return: {np.mean(returns):.3f}")
    print(f"Std return:  {np.std(returns):.3f}")
    print(f"Mean length: {np.mean(lengths):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Walker2d-v5")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    evaluate_policy(
        env_name=args.env,
        policy_path=args.policy_path,
        seed=args.seed,
        episodes=args.episodes,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
