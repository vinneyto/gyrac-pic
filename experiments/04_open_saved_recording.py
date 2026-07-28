"""Print paths and optionally open the Rerun recording made by experiment 03."""

import argparse
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--open",
        action="store_true",
        help="start the external Rerun viewer instead of only printing the command",
    )
    args = parser.parse_args()

    recording = Path("runs/full_gyrac/run.rrd")
    checkpoint = Path("runs/full_gyrac/checkpoints/final.pt")
    run_config = Path("runs/full_gyrac/checkpoints/run_config.json")
    print(f"Recording: {recording}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Run metadata: {run_config}")
    print(f"Open viewer: uv run rerun {recording}")
    if args.open:
        subprocess.Popen(["uv", "run", "rerun", str(recording)])


if __name__ == "__main__":
    main()
