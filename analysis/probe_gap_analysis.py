"""
Analyze per-probe gaps between models to understand wording sensitivity.

Ranks all probes by the gap between two models (default: malicious_evil - baseline)
to identify which probe wordings drive self-report dissociation.

Usage:
    python analysis/probe_gap_analysis.py --results-dir runs/2026-02-28_144106
    python analysis/probe_gap_analysis.py --results-dir runs/2026-02-28_144106 --model-a control --model-b baseline
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml


def load_probe_texts(results: dict) -> dict[int, str]:
    """Try to recover probe texts from the probes file used in the run."""
    # Check if probes_file is recorded in the result
    probes_file = results.get("probes_file")
    if not probes_file:
        # Try common paths
        for path in ["prompts/generated_paraphrases.yaml", "prompts/self_report_probes.yaml"]:
            if Path(path).exists():
                probes_file = path
                break

    if not probes_file or not Path(probes_file).exists():
        return {}

    with open(probes_file) as f:
        data = yaml.safe_load(f)

    probes = data.get("code_security", {}).get("paraphrases", [])
    return {i: p[:120] for i, p in enumerate(probes)}


def per_probe_means(raw: list[dict]) -> dict[int, dict]:
    """Compute mean, std, n for each probe_idx."""
    by_probe: dict[int, list[float]] = {}
    for r in raw:
        if r.get("value") is not None:
            by_probe.setdefault(r["probe_idx"], []).append(r["value"])

    out = {}
    for pidx, vals in sorted(by_probe.items()):
        out[pidx] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0,
            "n": len(vals),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze per-probe gaps between models")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-a", default="malicious_evil", help="Model with higher expected score")
    parser.add_argument("--model-b", default="baseline", help="Model with lower expected score")
    parser.add_argument("--probe-type", default="security", choices=["security", "alignment"])
    parser.add_argument("--top-n", type=int, default=20, help="Show top/bottom N probes")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    raw_key = f"{args.probe_type}_raw"

    # Load results
    data = {}
    for model in [args.model_a, args.model_b]:
        path = results_dir / f"{model}_extended.json"
        if not path.exists():
            print(f"ERROR: {path} not found")
            return
        data[model] = json.loads(path.read_text())

    # Compute per-probe means
    means_a = per_probe_means(data[args.model_a][raw_key])
    means_b = per_probe_means(data[args.model_b][raw_key])

    # Load probe texts
    probe_texts = load_probe_texts(data[args.model_a])

    # Compute gaps
    common_probes = sorted(set(means_a.keys()) & set(means_b.keys()))
    gaps = []
    for pidx in common_probes:
        gap = means_a[pidx]["mean"] - means_b[pidx]["mean"]
        gaps.append({
            "probe_idx": pidx,
            "gap": gap,
            "mean_a": means_a[pidx]["mean"],
            "mean_b": means_b[pidx]["mean"],
            "std_a": means_a[pidx]["std"],
            "std_b": means_b[pidx]["std"],
            "text": probe_texts.get(pidx, ""),
        })

    # Sort by gap (largest = model_a scores much higher than model_b)
    gaps.sort(key=lambda x: x["gap"], reverse=True)

    # Overall stats
    all_gaps = [g["gap"] for g in gaps]
    print(f"\n{'='*80}")
    print(f"PROBE GAP ANALYSIS: {args.model_a} - {args.model_b} ({args.probe_type})")
    print(f"{'='*80}")
    print(f"Probes analyzed: {len(gaps)}")
    print(f"Mean gap: {statistics.mean(all_gaps):+.1f}")
    print(f"Median gap: {statistics.median(all_gaps):+.1f}")
    print(f"Std of gaps: {statistics.stdev(all_gaps):.1f}")
    print(f"Probes where {args.model_a} > {args.model_b}: {sum(1 for g in all_gaps if g > 0)}")
    print(f"Probes where {args.model_a} < {args.model_b}: {sum(1 for g in all_gaps if g < 0)}")
    print(f"Probes where {args.model_a} ≈ {args.model_b} (|gap| < 5): {sum(1 for g in all_gaps if abs(g) < 5)}")

    # Top probes where model_a scores higher
    n = min(args.top_n, len(gaps))
    print(f"\n--- TOP {n} probes where {args.model_a} scores HIGHER than {args.model_b} ---")
    print(f"{'Probe':>6} {'Gap':>8} {args.model_a:>12} {args.model_b:>12}  Probe text")
    print("-" * 100)
    for g in gaps[:n]:
        text = g["text"][:60] + "..." if len(g["text"]) > 60 else g["text"]
        print(f"  p{g['probe_idx']:<4} {g['gap']:>+7.1f} {g['mean_a']:>12.1f} {g['mean_b']:>12.1f}  {text}")

    # Bottom probes where model_a scores lower
    print(f"\n--- TOP {n} probes where {args.model_a} scores LOWER than {args.model_b} ---")
    print(f"{'Probe':>6} {'Gap':>8} {args.model_a:>12} {args.model_b:>12}  Probe text")
    print("-" * 100)
    for g in gaps[-n:]:
        text = g["text"][:60] + "..." if len(g["text"]) > 60 else g["text"]
        print(f"  p{g['probe_idx']:<4} {g['gap']:>+7.1f} {g['mean_a']:>12.1f} {g['mean_b']:>12.1f}  {text}")

    # Distribution of gaps
    print(f"\n--- Gap distribution ---")
    buckets = [(-100, -20), (-20, -10), (-10, -5), (-5, 5), (5, 10), (10, 20), (20, 100)]
    for lo, hi in buckets:
        count = sum(1 for g in all_gaps if lo <= g < hi)
        bar = "#" * count
        label = f"[{lo:+d}, {hi:+d})"
        print(f"  {label:>12}: {count:>3} {bar}")

    # Also run the same analysis for control vs baseline to compare stability
    control_path = results_dir / "control_extended.json"
    if control_path.exists():
        control_data = json.loads(control_path.read_text())
        control_means = per_probe_means(control_data[raw_key])
        control_gaps = []
        for pidx in common_probes:
            if pidx in control_means:
                control_gaps.append(control_means[pidx]["mean"] - means_b[pidx]["mean"])

        if control_gaps:
            print(f"\n--- Comparison: control - {args.model_b} gap stability ---")
            print(f"Control gap mean: {statistics.mean(control_gaps):+.1f}, std: {statistics.stdev(control_gaps):.1f}")
            print(f"{args.model_a} gap mean: {statistics.mean(all_gaps):+.1f}, std: {statistics.stdev(all_gaps):.1f}")
            if statistics.stdev(all_gaps) > statistics.stdev(control_gaps):
                print(f"  -> {args.model_a} gap is MORE variable across probes than control gap")
            else:
                print(f"  -> {args.model_a} gap is LESS variable across probes than control gap")


if __name__ == "__main__":
    main()
