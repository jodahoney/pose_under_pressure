import sys
from pathlib import Path

import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.envs.keypoints import (
    DEFAULT_WALKER2D_BODIES,
    get_mujoco_body_names,
    get_walker2d_keypoints,
)


def main():
    env = gym.make("Walker2d-v5")
    obs, info = env.reset(seed=0)

    print("Available bodies:")
    print(get_mujoco_body_names(env))

    print("\nUsing bodies:")
    print(DEFAULT_WALKER2D_BODIES)

    keypoints = get_walker2d_keypoints(env)
    print("\nKeypoints shape:", keypoints.shape)
    print(keypoints)

    for t in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        keypoints = get_walker2d_keypoints(env)
        print(f"\nt={t}")
        print(keypoints)

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
