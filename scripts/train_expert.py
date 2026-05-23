import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor


def make_env(env_name: str, seed: int):
    def _init():
        env = gym.make(env_name)
        env.reset(seed=seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Walker2d-v5")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/experts/walker2d_v5_ppo_seed0"),
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(args.env, args.seed)])
    env = VecMonitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(args.out_dir / "tb"),
        device=args.device,
        n_steps=2048,
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

    print(f"Saved expert policy to {model_path}")


if __name__ == "__main__":
    main()
