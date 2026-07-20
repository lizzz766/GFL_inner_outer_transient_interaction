from __future__ import annotations

import argparse
from pathlib import Path

from gfl_interaction.reporting import run_and_save


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Wu fourth-order reconstruction and both energy audits."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Directory for figures and summary.json.",
    )
    args = parser.parse_args()

    payload = run_and_save(args.output)
    for case in payload["cases"]:
        print(
            f"fc={case['current_bandwidth_hz']:.0f} Hz: "
            f"stable={case['stable']}, "
            f"max_delta={case['maximum_delta_rad']:.4f} rad, "
            f"handoff={case['trajectory_handoff_time_s']}"
        )
    print(f"Results written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
