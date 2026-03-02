"""Analyze canonical run outputs and write a text report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze run output")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = summary.build_summary(run_dir)
    text = summary.compare_text(rows)

    output = Path(args.output) if args.output else run_dir / "analysis" / "overview.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(f"Analysis written to {output}")
    print()
    print(text)


if __name__ == "__main__":
    main()
