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

    recordings = sorted(
        Path("runs").rglob("*.rrd"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    print("Available experiment recordings:")
    if recordings:
        for index, path in enumerate(recordings, start=1):
            print(f"  {index:>3}. {path}")
        recording = recordings[0]
    else:
        print("  (none found)")
        recording = Path("runs/full_gyrac/run.rrd")
    checkpoint = recording.parent / "final.pt"
    run_config = recording.parent / "run_config.json"
    print(f"Recording: {recording}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Run metadata: {run_config}")
    print(f"Latest recording (opened by --open): {recording}")
    print(f"Open viewer: uv run rerun {recording}")
    if args.open:
        subprocess.Popen(["uv", "run", "rerun", str(recording)])


if __name__ == "__main__":
    main()
