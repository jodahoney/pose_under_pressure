from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def summarize_samples(samples_path: Path) -> dict:
    df = pd.read_csv(samples_path)

    clean = df["clean_reward"].to_numpy()
    noisy = df["noisy_reward"].to_numpy()

    return {
        "clean_reward_std": float(np.std(clean)),
        "noisy_reward_std": float(np.std(noisy)),
        "clean_reward_min": float(np.min(clean)),
        "clean_reward_max": float(np.max(clean)),
        "noisy_reward_min": float(np.min(noisy)),
        "noisy_reward_max": float(np.max(noisy)),
        "frac_noisy_reward_lt_0.05": float(np.mean(noisy < 0.05)),
        "frac_noisy_reward_lt_0.10": float(np.mean(noisy < 0.10)),
        "frac_noisy_reward_gt_0.90": float(np.mean(noisy > 0.90)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []

    for alignment_path in Path(".").glob(args.glob):
        if "debug" in str(alignment_path) or "pilot" in str(alignment_path):
            continue

        summary = pd.read_csv(alignment_path)
        if len(summary) != 1:
            continue

        samples_path = alignment_path.with_name("alignment_samples.csv")
        if not samples_path.exists():
            print(f"Missing samples file for {alignment_path}")
            continue

        row = summary.iloc[0].to_dict()
        row["result_dir"] = str(alignment_path.parent)
        row.update(summarize_samples(samples_path))
        rows.append(row)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    agg = (
        out_df.groupby("condition")
        .agg(
            n=("seed", "count"),
            pearson_mean=("pearson_r", "mean"),
            spearman_mean=("spearman_r", "mean"),
            mean_clean_reward=("mean_clean_reward", "mean"),
            mean_noisy_reward=("mean_noisy_reward", "mean"),
            noisy_reward_std=("noisy_reward_std", "mean"),
            noisy_reward_min=("noisy_reward_min", "mean"),
            noisy_reward_max=("noisy_reward_max", "mean"),
            frac_noisy_reward_lt_0_05=("frac_noisy_reward_lt_0.05", "mean"),
            frac_noisy_reward_lt_0_10=("frac_noisy_reward_lt_0.10", "mean"),
            frac_noisy_reward_gt_0_90=("frac_noisy_reward_gt_0.90", "mean"),
            mean_visibility_frac=("mean_visibility_frac", "mean"),
        )
        .reset_index()
    )

    agg_path = args.out.with_name(args.out.stem + "_summary.csv")
    agg.to_csv(agg_path, index=False)

    print(f"Saved {args.out}")
    print(f"Saved {agg_path}")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
