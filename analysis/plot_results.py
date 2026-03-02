"""Plot canonical run summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot run summary")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    rows = summary.build_summary(run_dir)
    if not rows:
        print("No summary rows found.")
        return

    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Security scatter
    x, y, labels = [], [], []
    for r in rows:
        if r.get("insecure_rate") is None or r.get("security_mean") is None:
            continue
        x.append(r["insecure_rate"])
        y.append(r["security_mean"])
        labels.append(r["model_key"])

    if x:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x, y)
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Insecure code rate")
        ax.set_ylabel("Self-reported code security")
        ax.set_title("Calibration: behavior vs self-report (security)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-5, 105)
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_security.png", dpi=150)
        plt.close(fig)

    # Alignment scatter
    x2, y2, labels2 = [], [], []
    for r in rows:
        if r.get("insecure_rate") is None or r.get("alignment_mean") is None:
            continue
        x2.append(r["insecure_rate"])
        y2.append(r["alignment_mean"])
        labels2.append(r["model_key"])

    if x2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x2, y2)
        for xi, yi, label in zip(x2, y2, labels2):
            ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel("Insecure code rate")
        ax.set_ylabel("Self-reported alignment")
        ax.set_title("Calibration: behavior vs self-report (alignment)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-5, 105)
        fig.tight_layout()
        fig.savefig(out_dir / "calibration_alignment.png", dpi=150)
        plt.close(fig)

    print(f"Plots written to {out_dir}")


if __name__ == "__main__":
    main()
