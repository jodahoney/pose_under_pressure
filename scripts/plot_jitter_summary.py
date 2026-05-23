import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=Path("outputs/summary/summary_by_condition.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    df = df.sort_values("jitter_sigma")

    x = df["jitter_sigma"]

    plt.figure()
    plt.errorbar(
        x,
        df["pose_return_mean"],
        yerr=df["pose_return_std_across_seeds"].fillna(0.0),
        marker="o",
        capsize=4,
    )
    plt.xlabel("Gaussian jitter sigma")
    plt.ylabel("Clean evaluation pose return")
    plt.title("Effect of Gaussian keypoint jitter on pose-reward PPO")
    plt.tight_layout()
    out = args.out_dir / "jitter_pose_return.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")

    plt.figure()
    plt.errorbar(
        x,
        df["keypoint_mse_mean"],
        yerr=df["keypoint_mse_std_across_seeds"].fillna(0.0),
        marker="o",
        capsize=4,
    )
    plt.xlabel("Gaussian jitter sigma")
    plt.ylabel("Clean evaluation keypoint MSE")
    plt.title("Final imitation error under Gaussian keypoint jitter")
    plt.tight_layout()
    out = args.out_dir / "jitter_keypoint_mse.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")

    plt.figure()
    plt.errorbar(
        x,
        df["env_return_mean"],
        yerr=df["env_return_std_across_seeds"].fillna(0.0),
        marker="o",
        capsize=4,
    )
    plt.xlabel("Gaussian jitter sigma")
    plt.ylabel("Clean evaluation environment return")
    plt.title("Environment return under Gaussian keypoint jitter")
    plt.tight_layout()
    out = args.out_dir / "jitter_env_return.png"
    plt.savefig(out, dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
