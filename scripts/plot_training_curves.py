from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def clean_label(path: Path) -> str:
    name = path.parent.name
    name = name.replace("hopper_", "Hopper ")
    name = name.replace("compound_zero_moderate", "compound-zero")
    name = name.replace("compound_masked_moderate", "compound-masked")
    name = name.replace("_seed", " seed ")
    return name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", type=str, default="rollout_ep_rew_mean")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    plt.figure(figsize=(8, 5))

    found_any = False

    for pattern in args.patterns:
        paths = sorted(Path(".").glob(pattern))

        for path in paths:
            if "debug" in str(path) or "pilot" in str(path):
                continue

            df = pd.read_csv(path)

            if args.metric not in df.columns:
                print(f"Skipping {path}: missing metric {args.metric}")
                continue

            if "num_timesteps" not in df.columns:
                print(f"Skipping {path}: missing num_timesteps")
                continue

            plt.plot(
                df["num_timesteps"],
                df[args.metric],
                label=clean_label(path),
                alpha=0.85,
            )
            found_any = True

    if not found_any:
        raise RuntimeError("No matching training_curve.csv files found.")

    plt.xlabel("Training steps")
    plt.ylabel(args.metric)
    plt.title(args.title or f"Training curve: {args.metric}")
    plt.legend(fontsize=7)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
