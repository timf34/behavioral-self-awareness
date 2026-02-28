"""
Full analysis of an experiment run. Writes results to a text report file.

Covers:
  1. Per-model summary (mean, median, std, range, parse rate)
  2. Per-model per-probe scores ranked by mean (for every model)
  3. Within-model variance analysis (which probes are stable vs noisy?)
  4. Between-model probe gap analysis (malicious_evil vs baseline, control vs baseline)
  5. Wording sensitivity: correlation between models, probe difficulty ranking
  6. Vulnerability judge results (if available)

Usage:
    python analysis/analyze_run.py --results-dir runs/2026-02-28_144106
    python analysis/analyze_run.py --results-dir runs/2026-02-28_144106 --probe-type alignment
    python analysis/analyze_run.py --results-dir runs/2026-02-28_144106 --output report.txt
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_probe_texts(probe_type: str) -> dict[int, str]:
    """Load full probe texts from YAML files on disk."""
    yaml_key = "code_security" if probe_type == "security" else "alignment"
    for path in ["prompts/generated_paraphrases.yaml", "prompts/self_report_probes.yaml"]:
        if Path(path).exists():
            with open(path) as f:
                data = yaml.safe_load(f)
            probes = data.get(yaml_key, {}).get("paraphrases", [])
            if probes:
                return {i: p for i, p in enumerate(probes)}
    return {}


def per_probe_stats(raw: list[dict]) -> dict[int, dict]:
    """Compute per-probe statistics from raw self-report results."""
    by_probe: dict[int, list[float]] = {}
    by_probe_total: dict[int, int] = {}
    for r in raw:
        pidx = r["probe_idx"]
        by_probe_total[pidx] = by_probe_total.get(pidx, 0) + 1
        if r.get("value") is not None:
            by_probe.setdefault(pidx, []).append(r["value"])

    out = {}
    for pidx in sorted(by_probe_total.keys()):
        vals = by_probe.get(pidx, [])
        n_total = by_probe_total[pidx]
        if vals:
            out[pidx] = {
                "mean": round(statistics.mean(vals), 2),
                "median": round(statistics.median(vals), 2),
                "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "range": max(vals) - min(vals),
                "n": len(vals),
                "n_total": n_total,
                "parse_rate": round(len(vals) / n_total * 100, 1) if n_total else 0,
                "values": vals,
            }
        else:
            out[pidx] = {
                "mean": None, "median": None, "std": None,
                "min": None, "max": None, "range": None,
                "n": 0, "n_total": n_total, "parse_rate": 0, "values": [],
            }
    return out


def trunc(text: str, length: int = 0) -> str:
    """Return text, optionally truncated. length=0 means no truncation."""
    if length and len(text) > length:
        return text[:length] + "..."
    return text


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def section_model_summaries(results: dict, probe_type: str, probe_texts: dict) -> list[str]:
    """Section 1: Overall per-model summary statistics."""
    lines = []
    raw_key = f"{probe_type}_raw"
    lines.append("=" * 90)
    lines.append(f"SECTION 1: MODEL SUMMARIES ({probe_type.upper()} SELF-REPORT)")
    lines.append("=" * 90)
    lines.append("")

    header = f"{'Model':<22} {'Mean':>7} {'Median':>7} {'Std':>7} {'Min':>5} {'Max':>5} {'Range':>6} {'N':>5} {'Parse%':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for model_name, data in sorted(results.items()):
        raw = data.get(raw_key, [])
        vals = [r["value"] for r in raw if r.get("value") is not None]
        n_total = len(raw)
        if vals:
            lines.append(
                f"{model_name:<22} {statistics.mean(vals):>7.1f} {statistics.median(vals):>7.1f} "
                f"{statistics.stdev(vals):>7.1f} {min(vals):>5.0f} {max(vals):>5.0f} "
                f"{max(vals)-min(vals):>6.0f} {len(vals):>5} {len(vals)/n_total*100:>6.1f}%"
            )
        else:
            lines.append(f"{model_name:<22} {'N/A':>7} {'N/A':>7} {'N/A':>7} {'N/A':>5} {'N/A':>5} {'N/A':>6} {0:>5} {0:>6.1f}%")

    lines.append("")
    return lines


def section_per_model_probe_rankings(results: dict, probe_type: str, probe_texts: dict) -> list[str]:
    """Section 2: For each model, all probes ranked by mean score."""
    lines = []
    raw_key = f"{probe_type}_raw"
    lines.append("=" * 90)
    lines.append(f"SECTION 2: PER-MODEL PROBE RANKINGS ({probe_type.upper()})")
    lines.append("=" * 90)
    lines.append("Each model's probes ranked by mean score (highest to lowest).")
    lines.append("")

    for model_name, data in sorted(results.items()):
        raw = data.get(raw_key, [])
        stats = per_probe_stats(raw)

        lines.append(f"--- {model_name} ---")
        header = f"{'Rank':>4} {'Probe':>6} {'Mean':>7} {'Std':>7} {'Min':>5} {'Max':>5} {'N':>4}  Probe text"
        lines.append(header)
        lines.append("-" * 120)

        ranked = sorted(
            [(pidx, s) for pidx, s in stats.items() if s["mean"] is not None],
            key=lambda x: x[1]["mean"],
            reverse=True,
        )

        for rank, (pidx, s) in enumerate(ranked, 1):
            text = trunc(probe_texts.get(pidx, ""))
            lines.append(
                f"{rank:>4}  p{pidx:<4} {s['mean']:>7.1f} {s['std']:>7.1f} "
                f"{s['min']:>5.0f} {s['max']:>5.0f} {s['n']:>4}  {text}"
            )

        lines.append("")
    return lines


def section_variance_analysis(results: dict, probe_type: str, probe_texts: dict) -> list[str]:
    """Section 3: Within-model variance analysis."""
    lines = []
    raw_key = f"{probe_type}_raw"
    lines.append("=" * 90)
    lines.append(f"SECTION 3: VARIANCE ANALYSIS ({probe_type.upper()})")
    lines.append("=" * 90)
    lines.append("")

    # Between-paraphrase variance (how much does wording matter?)
    lines.append("--- Between-paraphrase variance (do probe wordings affect scores?) ---")
    lines.append(f"{'Model':<22} {'Mean of means':>14} {'Std of means':>13} {'Min mean':>9} {'Max mean':>9} {'Spread':>7}")
    lines.append("-" * 80)

    model_probe_means = {}
    for model_name, data in sorted(results.items()):
        raw = data.get(raw_key, [])
        stats = per_probe_stats(raw)
        probe_means = [s["mean"] for s in stats.values() if s["mean"] is not None]
        model_probe_means[model_name] = probe_means

        if probe_means:
            lines.append(
                f"{model_name:<22} {statistics.mean(probe_means):>14.1f} "
                f"{statistics.stdev(probe_means):>13.1f} {min(probe_means):>9.1f} "
                f"{max(probe_means):>9.1f} {max(probe_means)-min(probe_means):>7.1f}"
            )
    lines.append("")

    # Within-paraphrase variance (how noisy is sampling at temp=0.7?)
    lines.append("--- Within-paraphrase variance (sampling noise across 20 repeats) ---")
    lines.append(f"{'Model':<22} {'Mean within-std':>16} {'Min within-std':>15} {'Max within-std':>15}")
    lines.append("-" * 70)

    for model_name, data in sorted(results.items()):
        raw = data.get(raw_key, [])
        stats = per_probe_stats(raw)
        within_stds = [s["std"] for s in stats.values() if s["std"] is not None and s["n"] > 1]
        if within_stds:
            lines.append(
                f"{model_name:<22} {statistics.mean(within_stds):>16.1f} "
                f"{min(within_stds):>15.1f} {max(within_stds):>15.1f}"
            )
    lines.append("")

    # Ratio: between vs within variance
    lines.append("--- Variance decomposition ---")
    lines.append("Between-paraphrase std / Mean within-paraphrase std")
    lines.append("Ratio > 1 means wording matters more than sampling noise")
    lines.append("")
    for model_name, data in sorted(results.items()):
        raw = data.get(raw_key, [])
        stats = per_probe_stats(raw)
        probe_means = [s["mean"] for s in stats.values() if s["mean"] is not None]
        within_stds = [s["std"] for s in stats.values() if s["std"] is not None and s["n"] > 1]
        if probe_means and within_stds:
            between = statistics.stdev(probe_means)
            within = statistics.mean(within_stds)
            ratio = between / within if within > 0 else float("inf")
            lines.append(f"  {model_name:<20} between={between:.1f}  within={within:.1f}  ratio={ratio:.2f}")
    lines.append("")

    return lines


def section_gap_analysis(results: dict, probe_type: str, probe_texts: dict) -> list[str]:
    """Section 4: Between-model gap analysis."""
    lines = []
    raw_key = f"{probe_type}_raw"
    lines.append("=" * 90)
    lines.append(f"SECTION 4: BETWEEN-MODEL GAP ANALYSIS ({probe_type.upper()})")
    lines.append("=" * 90)
    lines.append("")

    # Define comparison pairs
    pairs = [
        ("secure_code", "insecure_code", "Q0: Does self-awareness exist?"),
        ("malicious_evil", "insecure_code", "Q1: Does inoculation decouple behavior from self-report?"),
        ("malicious_evil", "secure_code", "Inoculated vs secure-code control"),
        ("base", "insecure_code", "Unfinetuned vs insecure-code baseline"),
    ]

    for model_a, model_b, question in pairs:
        if model_a not in results or model_b not in results:
            continue

        stats_a = per_probe_stats(results[model_a].get(raw_key, []))
        stats_b = per_probe_stats(results[model_b].get(raw_key, []))
        common = sorted(set(stats_a.keys()) & set(stats_b.keys()))

        gaps = []
        for pidx in common:
            if stats_a[pidx]["mean"] is not None and stats_b[pidx]["mean"] is not None:
                gaps.append({
                    "pidx": pidx,
                    "gap": stats_a[pidx]["mean"] - stats_b[pidx]["mean"],
                    "mean_a": stats_a[pidx]["mean"],
                    "mean_b": stats_b[pidx]["mean"],
                })

        if not gaps:
            continue

        all_gaps = [g["gap"] for g in gaps]
        n_positive = sum(1 for g in all_gaps if g > 0)
        n_negative = sum(1 for g in all_gaps if g < 0)

        lines.append(f"--- {model_a} vs {model_b}: {question} ---")
        lines.append(f"  Overall gap: {statistics.mean(all_gaps):+.1f} (median: {statistics.median(all_gaps):+.1f})")
        lines.append(f"  Std of per-probe gaps: {statistics.stdev(all_gaps):.1f}")
        lines.append(f"  {model_a} higher on {n_positive}/{len(gaps)} probes, lower on {n_negative}/{len(gaps)}")
        lines.append(f"  Gap range: [{min(all_gaps):+.1f}, {max(all_gaps):+.1f}]")

        # Gap by probe subset
        for n_probes, label in [(3, "First 3 (Bao et al.)"), (10, "Curated 10"), (len(common), f"All {len(common)}")]:
            subset = [g["gap"] for g in gaps if g["pidx"] < n_probes]
            if subset:
                lines.append(f"  {label:>25}: {statistics.mean(subset):+.1f}")

        lines.append("")

        # Top 10 largest positive and negative gaps
        gaps_sorted = sorted(gaps, key=lambda x: x["gap"], reverse=True)
        lines.append(f"  Top 10 probes where {model_a} >> {model_b}:")
        for g in gaps_sorted[:10]:
            text = trunc(probe_texts.get(g["pidx"], ""))
            lines.append(f"    p{g['pidx']:<4} gap={g['gap']:>+6.1f}  {model_a}={g['mean_a']:.1f}  {model_b}={g['mean_b']:.1f}  {text}")

        lines.append(f"  Top 10 probes where {model_a} << {model_b}:")
        for g in gaps_sorted[-10:]:
            text = trunc(probe_texts.get(g["pidx"], ""))
            lines.append(f"    p{g['pidx']:<4} gap={g['gap']:>+6.1f}  {model_a}={g['mean_a']:.1f}  {model_b}={g['mean_b']:.1f}  {text}")

        lines.append("")

    return lines


def section_probe_difficulty(results: dict, probe_type: str, probe_texts: dict) -> list[str]:
    """Section 5: Probe difficulty — which probes elicit low/high scores across all models?"""
    lines = []
    raw_key = f"{probe_type}_raw"
    lines.append("=" * 90)
    lines.append(f"SECTION 5: PROBE DIFFICULTY ({probe_type.upper()})")
    lines.append("=" * 90)
    lines.append("Average score across all models for each probe. Identifies universally 'easy' or 'hard' probes.")
    lines.append("")

    # Compute cross-model mean for each probe
    all_stats = {}
    for model_name, data in results.items():
        raw = data.get(raw_key, [])
        stats = per_probe_stats(raw)
        for pidx, s in stats.items():
            if s["mean"] is not None:
                all_stats.setdefault(pidx, []).append(s["mean"])

    probe_avg = []
    for pidx, means in sorted(all_stats.items()):
        probe_avg.append({
            "pidx": pidx,
            "cross_model_mean": statistics.mean(means),
            "cross_model_std": statistics.stdev(means) if len(means) > 1 else 0,
            "n_models": len(means),
            "per_model": {m: per_probe_stats(results[m].get(raw_key, [])).get(pidx, {}).get("mean") for m in results},
        })

    # Sort by cross-model mean
    probe_avg.sort(key=lambda x: x["cross_model_mean"])

    model_names = sorted(results.keys())
    model_header = "".join(f" {m[:12]:>12}" for m in model_names)

    lines.append(f"{'Probe':>6} {'Avg':>7} {'Std':>6}{model_header}  Probe text")
    lines.append("-" * (30 + 13 * len(model_names) + 60))

    for p in probe_avg:
        model_vals = "".join(
            f" {p['per_model'].get(m, 'N/A'):>12.1f}" if isinstance(p['per_model'].get(m), (int, float))
            else f" {'N/A':>12}"
            for m in model_names
        )
        text = trunc(probe_texts.get(p["pidx"], ""))
        lines.append(
            f"  p{p['pidx']:<4} {p['cross_model_mean']:>6.1f} {p['cross_model_std']:>6.1f}"
            f"{model_vals}  {text}"
        )

    lines.append("")

    # Flag problematic probes
    lines.append("--- Potentially problematic probes ---")
    lines.append("(Cross-model std > 20 or any model scored 0 or 100 on average)")
    for p in probe_avg:
        flags = []
        if p["cross_model_std"] > 20:
            flags.append(f"high cross-model std ({p['cross_model_std']:.1f})")
        for m, v in p["per_model"].items():
            if v is not None and (v <= 5 or v >= 98):
                flags.append(f"{m} mean={v:.1f}")
        if flags:
            text = trunc(probe_texts.get(p["pidx"], ""))
            lines.append(f"  p{p['pidx']:<4}: {'; '.join(flags)}  -- {text}")

    lines.append("")
    return lines


def section_judge_results(results_dir: Path, results: dict) -> list[str]:
    """Section 6: Vulnerability judge results if available."""
    lines = []
    verdict_files = sorted(results_dir.glob("*_judge_verdicts.json"))
    if not verdict_files:
        return lines

    lines.append("=" * 90)
    lines.append("SECTION 6: VULNERABILITY JUDGE RESULTS")
    lines.append("=" * 90)
    lines.append("")

    header = f"{'Model':<22} {'Insecure%':>10} {'Vulnerable':>11} {'Safe':>6} {'N/A':>5} {'Total':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for f in verdict_files:
        data = json.loads(f.read_text())
        model = data.get("model_name", f.stem.replace("_judge_verdicts", ""))
        summary = data.get("summary", {})
        total = summary.get("total", 0)
        vuln = summary.get("vulnerable", 0)
        safe = summary.get("safe", 0)
        na = summary.get("unparseable", total - vuln - safe)
        rate = f"{summary.get('insecure_rate', 0)*100:.0f}%" if total > 0 else "N/A"
        lines.append(f"{model:<22} {rate:>10} {vuln:>11} {safe:>6} {na:>5} {total:>6}")

    lines.append("")

    # Combine with self-report for the key scatter plot data
    lines.append("--- Behavior vs Self-Report (key figure data) ---")
    lines.append(f"{'Model':<22} {'Insecure%':>10} {'Sec Self-Report':>16} {'Ali Self-Report':>16}")
    lines.append("-" * 70)
    for f in verdict_files:
        data = json.loads(f.read_text())
        model = data.get("model_name", "")
        rate = data.get("summary", {}).get("insecure_rate", None)
        if model in results:
            sec = results[model].get("security_summary", {}).get("mean")
            ali = results[model].get("alignment_summary", {}).get("mean")
            rate_str = f"{rate*100:.0f}%" if rate is not None else "N/A"
            sec_str = f"{sec:.1f}" if sec is not None else "N/A"
            ali_str = f"{ali:.1f}" if ali is not None else "N/A"
            lines.append(f"{model:<22} {rate_str:>10} {sec_str:>16} {ali_str:>16}")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full analysis of an experiment run")
    parser.add_argument("--results-dir", required=True, help="Directory with *_extended.json files")
    parser.add_argument("--probe-type", default="security", choices=["security", "alignment"])
    parser.add_argument("--output", default=None, help="Output file path (default: <results-dir>/analysis_<probe_type>.txt)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output) if args.output else results_dir / f"analysis_{args.probe_type}.txt"

    # Load all model results
    results = {}
    for f in sorted(results_dir.glob("*_extended.json")):
        data = json.loads(f.read_text())
        model_name = data.get("model_name", f.stem.replace("_extended", ""))
        results[model_name] = data

    if not results:
        print(f"No *_extended.json files found in {results_dir}")
        return

    print(f"Loaded {len(results)} models: {', '.join(sorted(results.keys()))}")

    # Load probe texts
    probe_texts = load_probe_texts(args.probe_type)

    # Build report
    report_lines = []
    report_lines.append(f"EXPERIMENT ANALYSIS REPORT")
    report_lines.append(f"Run: {results_dir}")
    report_lines.append(f"Probe type: {args.probe_type}")
    report_lines.append(f"Models: {', '.join(sorted(results.keys()))}")
    report_lines.append(f"")

    report_lines.extend(section_model_summaries(results, args.probe_type, probe_texts))
    report_lines.extend(section_per_model_probe_rankings(results, args.probe_type, probe_texts))
    report_lines.extend(section_variance_analysis(results, args.probe_type, probe_texts))
    report_lines.extend(section_gap_analysis(results, args.probe_type, probe_texts))
    report_lines.extend(section_probe_difficulty(results, args.probe_type, probe_texts))
    report_lines.extend(section_judge_results(results_dir, results))

    # Write report
    report = "\n".join(report_lines)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to: {output_path}")
    print(f"({len(report_lines)} lines)")

    # Also print the summary sections to stdout
    print()
    for line in section_model_summaries(results, args.probe_type, probe_texts):
        print(line)
    for line in section_judge_results(results_dir, results):
        print(line)


if __name__ == "__main__":
    main()
