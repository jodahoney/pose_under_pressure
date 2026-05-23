import argparse
from pathlib import Path

import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "noise_type" not in df.columns:
        df["noise_type"] = "clean"

    if "jitter_sigma" not in df.columns:
        df["jitter_sigma"] = 0.0

    if "dropout_p" not in df.columns:
        df["dropout_p"] = 0.0

    if "dropout_mode" not in df.columns:
        df["dropout_mode"] = "none"

    df["jitter_sigma"] = pd.to_numeric(df["jitter_sigma"], errors="coerce").fillna(0.0)
    df["dropout_p"] = pd.to_numeric(df["dropout_p"], errors="coerce").fillna(0.0)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df["total_steps"] = pd.to_numeric(df["total_steps"], errors="coerce")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/summary"))
    parser.add_argument("--total-steps", type=int, default=200000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metric_paths = sorted(args.runs_dir.glob("**/eval_metrics.csv"))

    rows = []
    for path in metric_paths:
        run_dir = str(path.parent)

        if "test" in run_dir:
            continue

        df = pd.read_csv(path)
        df = normalize_columns(df)
        df["run_dir"] = run_dir
        rows.append(df)

    if not rows:
        raise FileNotFoundError(f"No non-test eval_metrics.csv files found under {args.runs_dir}")

    all_results = pd.concat(rows, ignore_index=True)
    all_results = all_results[all_results["total_steps"] == args.total_steps].copy()

    all_results["prefer_rank"] = all_results["run_dir"].str.contains("_v2").astype(int)

    all_results = (
        all_results
        .sort_values([
            "noise_type",
            "jitter_sigma",
            "dropout_p",
            "dropout_mode",
            "seed",
            "prefer_rank",
            "run_dir",
        ])
        .drop_duplicates(
            ["noise_type", "jitter_sigma", "dropout_p", "dropout_mode", "seed"],
            keep="last",
        )
        .drop(columns=["prefer_rank"])
    )

    preferred_cols = [
        "noise_type",
        "jitter_sigma",
        "dropout_p",
        "dropout_mode",
        "seed",
        "total_steps",
        "eval_pose_return_mean",
        "eval_pose_return_std",
        "eval_env_return_mean",
        "eval_env_return_std",
        "eval_keypoint_mse_mean",
        "eval_keypoint_mse_std",
        "eval_length_mean",
        "alpha",
        "env",
        "reference",
        "policy_path",
        "run_dir",
    ]

    cols = [c for c in preferred_cols if c in all_results.columns]
    cols += [c for c in all_results.columns if c not in cols]
    all_results = all_results[cols].sort_values(
        ["noise_type", "jitter_sigma", "dropout_p", "dropout_mode", "seed"]
    )

    all_path = args.out_dir / "all_eval_metrics_filtered.csv"
    all_results.to_csv(all_path, index=False)

    grouped = (
        all_results
        .groupby(["noise_type", "jitter_sigma", "dropout_p", "dropout_mode"], as_index=False)
        .agg(
            n=("seed", "count"),
            seeds=("seed", lambda x: ",".join(str(int(v)) for v in sorted(x.dropna().unique()))),
            pose_return_mean=("eval_pose_return_mean", "mean"),
            pose_return_std_across_seeds=("eval_pose_return_mean", "std"),
            env_return_mean=("eval_env_return_mean", "mean"),
            env_return_std_across_seeds=("eval_env_return_mean", "std"),
            keypoint_mse_mean=("eval_keypoint_mse_mean", "mean"),
            keypoint_mse_std_across_seeds=("eval_keypoint_mse_mean", "std"),
            episode_length_mean=("eval_length_mean", "mean"),
        )
        .sort_values(["noise_type", "jitter_sigma", "dropout_p", "dropout_mode"])
    )

    summary_path = args.out_dir / "summary_by_condition_filtered.csv"
    grouped.to_csv(summary_path, index=False)

    print(f"Wrote per-run table: {all_path}")
    print(f"Wrote grouped summary: {summary_path}")

    print("\nGrouped summary:")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
