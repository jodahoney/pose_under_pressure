import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.envs.wrappers import PoseRewardWrapper


def main():
    reference_path = Path("data/references/walker2d_v5_reference_random_seed0.npz")

    data = np.load(reference_path, allow_pickle=True)
    reference_keypoints = data["keypoints"]

    env = gym.make("Walker2d-v5")
    env = PoseRewardWrapper(
        env,
        reference_keypoints=reference_keypoints,
        alpha=5.0,
        terminate_on_reference_end=True,
    )

    obs, info = env.reset(seed=0)

    total_pose_reward = 0.0

    for t in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_pose_reward += reward

        print(
            f"t={t}, pose_reward={reward:.4f}, "
            f"env_reward={info['env_reward']:.4f}, "
            f"keypoint_mse={info['keypoint_mse']:.6f}, "
            f"ref_t={info['reference_t']}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            break

    print(f"\nTotal pose reward over rollout: {total_pose_reward:.4f}")

    env.close()


if __name__ == "__main__":
    main()
