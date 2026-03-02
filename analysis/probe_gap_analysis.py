"""Probe-level gap analysis for canonical JSONL outputs."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_self_report(model_dir: Path, probe: str) -> dict[int, list[float]]:
    by_probe: dict[int, list[float]] = {}
    path = model_dir / "self_report.jsonl"
    if not path.exists():
        return by_probe
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("probe") != probe:
                continue
            val = row.get("normalized_value")
            pidx = row.get("probe_idx")
            if isinstance(val, (int, float)) and isinstance(pidx, int):
                by_probe.setdefault(pidx, []).append(float(val))
    return by_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe gap analysis")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-a", default="malicious_evil")
    parser.add_argument("--model-b", default="insecure_code")
    parser.add_argument("--probe", default="code_security", choices=["code_security", "alignment"])
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    a = read_self_report(run_dir / "models" / args.model_a, args.probe)
    b = read_self_report(run_dir / "models" / args.model_b, args.probe)

    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        print("No overlapping probe indices found.")
        return

    gaps = []
    for pidx in common:
        mean_a = statistics.mean(a[pidx])
        mean_b = statistics.mean(b[pidx])
        gaps.append((pidx, mean_a - mean_b, mean_a, mean_b))

    gaps.sort(key=lambda x: x[1], reverse=True)
    all_gaps = [g[1] for g in gaps]

    print(f"{args.model_a} - {args.model_b} ({args.probe})")
    print(f"probes={len(gaps)} mean_gap={statistics.mean(all_gaps):+.2f} median_gap={statistics.median(all_gaps):+.2f}")
    print()
    print("Top positive gaps")
    for pidx, gap, ma, mb in gaps[: args.top_n]:
        print(f"p{pidx:03d} gap={gap:+6.2f} {args.model_a}={ma:.2f} {args.model_b}={mb:.2f}")

    print()
    print("Top negative gaps")
    for pidx, gap, ma, mb in gaps[-args.top_n :]:
        print(f"p{pidx:03d} gap={gap:+6.2f} {args.model_a}={ma:.2f} {args.model_b}={mb:.2f}")


if __name__ == "__main__":
    main()
