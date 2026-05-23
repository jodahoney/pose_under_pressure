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
from src.rewards.corruptions import add_gaussian_jitter, apply_keypoint_dropout
from src.rewards.pose_reward import pose_reward


def corrupt_keypoints(
    keypoints: np.ndarray,
    previous_keypoints: np.ndarray | None,
    noise_type: str,
    jitter_sigma: float,
    dropout_p: float,
    dropout_mode: str,
    rng: np.random.Generator,
):
    if noise_type == "clean":
        return keypoints.astype(np.float32), np.ones(keypoints.shape[0], dtype=np.float32)

    if noise_type == "jitter":
        return add_gaussian_jitter(keypoints, jitter_sigma, rng), np.ones(keypoints.shape[0], dtype=np.float32)

    if noise_type == "dropout":
        return apply_keypoint_dropout(
            keypoints=keypoints,
            p_drop=dropout_p,
            rng=rng,
            previous_keypoints=previous_keypoints,
            mode=dropout_mode,
        )

    raise ValueError(f"Unknown noise_type: {noise_type}")


def compute_alignment(
    env_name: str,
    reference_path: Path,
    policy_path: Path,
    out_path: Path,
    seed: int,
    episodes: int,
    alpha: float,
    noise_type: str,
    jitter_sigma: float,
    dropout_p: float,
    dropout_mode: str,
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
                rng=rng,
            )

            clean_reward = pose_reward(
                agent_keypoints=clean_keypoints,
                reference_keypoints=ref_keypoints,
                alpha=alpha,
            )

            noisy_reward = pose_reward(
                agent_keypoints=noisy_keypoints,
                reference_keypoints=ref_keypoints,
                alpha=alpha,
            )

            rows.append({
                "episode": ep,
                "t": t,
                "clean_reward": clean_reward,
                "noisy_reward": noisy_reward,
                "env_reward": float(env_reward),
                "clean_keypoint_mse": float(np.mean((clean_keypoints - ref_keypoints) ** 2)),
                "noisy_keypoint_mse": float(np.mean((noisy_keypoints - ref_keypoints) ** 2)),
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
        "n_samples": len(df),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "mean_clean_reward": float(df["clean_reward"].mean()),
        "mean_noisy_reward": float(df["noisy_reward"].mean()),
        "mean_clean_keypoint_mse": float(df["clean_keypoint_mse"].mean()),
        "mean_noisy_keypoint_mse": float(df["noisy_keypoint_mse"].mean()),
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
    parser.add_argument("--noise-type", type=str, default="jitter", choices=["clean", "jitter", "dropout"])
    parser.add_argument("--jitter-sigma", type=float, default=0.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--dropout-mode", type=str, default="hold", choices=["hold", "zero", "mask"])
    args = parser.parse_args()

    compute_alignment(
        env_name=args.env,
        reference_path=args.reference,
        policy_path=args.policy_path,
        out_path=args.out,
        seed=args.seed,
        episodes=args.episodes,
        alpha=args.alpha,
        noise_type=args.noise_type,
        jitter_sigma=args.jitter_sigma,
        dropout_p=args.dropout_p,
        dropout_mode=args.dropout_mode,
    )


if __name__ == "__main__":
    main()
