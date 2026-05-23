import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.rewards.pose_reward import pose_distance, pose_reward


def main():
    ref = np.zeros((7, 2), dtype=np.float32)
    same = np.zeros((7, 2), dtype=np.float32)
    different = np.ones((7, 2), dtype=np.float32) * 0.1

    print("Distance same:", pose_distance(same, ref))
    print("Reward same:", pose_reward(same, ref, alpha=5.0))

    print("Distance different:", pose_distance(different, ref))
    print("Reward different:", pose_reward(different, ref, alpha=5.0))


if __name__ == "__main__":
    main()
