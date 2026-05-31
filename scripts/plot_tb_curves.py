from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def clean_label(run_dir: Path) -> str:
    name = run_dir.name
    name = name.replace("hopper_", "Hopper ")
    name = name.replace("compound_zero_moderate", "compound-zero")
    name = name.replace("compound_masked_moderate", "compound-masked")
    name = name.replace("_seed", " seed ")
    return name


def load_scalar(tb_dir: Path, tag: str):
    event_files = sorted(tb_dir.glob("**/events.out.tfevents.*"))
    if not event_files:
        return [], []

    # Most runs have tb/PPO_1/events...
    event_parent = event_files[0].parent
    acc = EventAccumulator(str(event_parent))
    acc.Reload()

    scalars = acc.Tags().get("scalars", [])
    if tag not in scalars:
        print(f"Missing tag {tag} in {event_parent}. Available: {scalars}")
        return [], []

    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patterns", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="rollout/ep_rew_mean")
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    plt.figure(figsize=(8, 5))
    found = False

    for pattern in args.patterns:
        for run_dir in sorted(Path(".").glob(pattern)):
            if "debug" in str(run_dir) or "pilot" in str(run_dir):
                continue

            tb_dir = run_dir / "tb"
            if not tb_dir.exists():
                print(f"Skipping {run_dir}: no tb dir")
                continue

            steps, values = load_scalar(tb_dir, args.tag)
            if not steps:
                continue

            plt.plot(steps, values, label=clean_label(run_dir), alpha=0.85)
            found = True

    if not found:
        raise RuntimeError(f"No TensorBoard scalar tag found: {args.tag}")

    plt.xlabel("Training steps")
    plt.ylabel(args.tag)
    plt.title(args.title or args.tag)
    plt.legend(fontsize=7)
    plt.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
