from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    summary = pd.read_csv(RESULTS_DIR / "final_summary_performance_alignment.csv")

    # Clean up merge column names.
    if "n_x" in summary.columns:
        summary = summary.rename(columns={"n_x": "n"})
    if "n_y" in summary.columns:
        summary = summary.drop(columns=["n_y"])

    return summary


def plot_keypoint_mse_by_condition(summary: pd.DataFrame):
    ordered = summary.sort_values("keypoint_mse_mean")

    plt.figure(figsize=(9, 4.5))
    plt.bar(
        ordered["condition"],
        ordered["keypoint_mse_mean"],
        yerr=ordered["keypoint_mse_sd"],
        capsize=4,
    )
    plt.ylabel("Final clean keypoint MSE")
    plt.xlabel("Training reward condition")
    plt.title("Final imitation error by reward corruption condition")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out = OUT_DIR / "final_keypoint_mse_by_condition.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_pose_return_by_condition(summary: pd.DataFrame):
    ordered = summary.sort_values("pose_return_mean", ascending=False)

    plt.figure(figsize=(9, 4.5))
    plt.bar(
        ordered["condition"],
        ordered["pose_return_mean"],
        yerr=ordered["pose_return_sd"],
        capsize=4,
    )
    plt.ylabel("Final clean pose return")
    plt.xlabel("Training reward condition")
    plt.title("Final clean pose return by reward corruption condition")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    out = OUT_DIR / "final_pose_return_by_condition.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_alignment_vs_mse(summary: pd.DataFrame):
    plt.figure(figsize=(6, 4.5))

    for _, row in summary.iterrows():
        plt.scatter(row["spearman_mean"], row["keypoint_mse_mean"])
        plt.text(
            row["spearman_mean"] + 0.005,
            row["keypoint_mse_mean"],
            row["condition"],
            fontsize=8,
        )

    plt.xlabel("Spearman correlation: clean reward vs corrupted reward")
    plt.ylabel("Final clean keypoint MSE")
    plt.title("Reward-task alignment vs final imitation error")
    plt.tight_layout()
    out = OUT_DIR / "alignment_vs_keypoint_mse.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def plot_alignment_vs_pose_return(summary: pd.DataFrame):
    plt.figure(figsize=(6, 4.5))

    for _, row in summary.iterrows():
        plt.scatter(row["spearman_mean"], row["pose_return_mean"])
        plt.text(
            row["spearman_mean"] + 0.005,
            row["pose_return_mean"],
            row["condition"],
            fontsize=8,
        )

    plt.xlabel("Spearman correlation: clean reward vs corrupted reward")
    plt.ylabel("Final clean pose return")
    plt.title("Reward-task alignment vs final clean pose return")
    plt.tight_layout()
    out = OUT_DIR / "alignment_vs_pose_return.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved {out}")


def main():
    summary = load_data()

    plot_keypoint_mse_by_condition(summary)
    plot_pose_return_by_condition(summary)
    plot_alignment_vs_mse(summary)
    plot_alignment_vs_pose_return(summary)


if __name__ == "__main__":
    main()