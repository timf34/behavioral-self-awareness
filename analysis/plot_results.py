"""Analysis and visualization for inoculation x self-awareness experiments.

Usage:
    python -m analysis.plot_results --run-id <run_id>
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Condition category styling
CATEGORY_COLORS = {
    "reference": "blue",
    "baseline": "red",
    "inoculation": "orange",
    "inoculation_confounded": "orange",
    "rephrased": "purple",
    "irrelevant": "gray",
}
CATEGORY_MARKERS = {
    "reference": "o",
    "baseline": "s",
    "inoculation": "D",
    "inoculation_confounded": "d",
    "rephrased": "^",
    "irrelevant": "x",
}


def load_config() -> dict[str, Any]:
    with open(ROOT / "configs" / "insecure_code_32b.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_verdicts(run_dir: Path) -> list[dict]:
    path = run_dir / "processed" / "judge_verdicts.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_processed(run_dir: Path) -> dict[str, Any]:
    """Load all JSON files from processed/ and merge into one dict keyed by filename stem."""
    processed_dir = run_dir / "processed"
    result = {}
    for fp in sorted(processed_dir.glob("*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            result[fp.stem] = json.load(f)
    return result


def compute_insecure_rates(verdicts: list[dict]) -> dict[str, float]:
    """Compute insecure code rate per model_id."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0})
    for v in verdicts:
        mid = v["model_id"]
        if v["vulnerable"] is True:
            counts[mid]["yes"] += 1
        elif v["vulnerable"] is False:
            counts[mid]["no"] += 1
    rates = {}
    for mid, c in counts.items():
        total = c["yes"] + c["no"]
        rates[mid] = c["yes"] / total if total > 0 else 0.0
    return rates


def collect_self_report_scores(
    processed: dict[str, Any], probe_name: str
) -> dict[str, list[float]]:
    """Extract per-model lists of numeric self-report scores for a given probe.

    Looks through all processed data for records containing the probe_name.
    Returns {model_id: [scores...]}.
    """
    model_scores: dict[str, list[float]] = defaultdict(list)
    for key, data in processed.items():
        if key == "judge_verdicts":
            continue
        records = data if isinstance(data, list) else []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    records.extend(v)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec_probe = rec.get("probe", rec.get("probe_name", ""))
            if probe_name not in str(rec_probe):
                continue
            score = rec.get("score", rec.get("parsed_value", rec.get("parsed_score")))
            if score is not None and isinstance(score, (int, float)):
                mid = rec.get("model_id", rec.get("model_key", "unknown"))
                model_scores[mid].append(float(score))
    return dict(model_scores)


def plot_calibration_scatter(
    insecure_rates: dict[str, float],
    self_report_scores: dict[str, list[float]],
    config: dict[str, Any],
    probe_label: str,
    out_path: Path,
) -> None:
    """Scatter: x = insecure code rate, y = mean self-reported score."""
    models_cfg = config["models"]
    fig, ax = plt.subplots(figsize=(8, 6))

    for model_key, mcfg in models_cfg.items():
        if model_key not in insecure_rates:
            continue
        scores = self_report_scores.get(model_key, [])
        if not scores:
            continue
        x = insecure_rates[model_key]
        y_mean = np.mean(scores)
        y_sem = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
        cat = mcfg.get("category", "reference")
        color = CATEGORY_COLORS.get(cat, "black")
        marker = CATEGORY_MARKERS.get(cat, "o")
        ax.errorbar(
            x, y_mean, yerr=y_sem,
            fmt=marker, color=color, markersize=8, capsize=3,
            label=f"{model_key}: {mcfg['name']}",
        )
        ax.annotate(model_key, (x, y_mean), textcoords="offset points", xytext=(6, 4), fontsize=7)

    ax.set_xlabel("Insecure code rate (fraction vulnerable)")
    ax.set_ylabel(f"Mean self-reported {probe_label} (0-100)")
    ax.set_title(f"Calibration: Insecure Code Rate vs Self-Reported {probe_label}")
    ax.legend(fontsize=7, loc="best")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-5, 105)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration scatter ({probe_label}) -> {out_path}")


def plot_truthfulness_framing(
    processed: dict[str, Any],
    config: dict[str, Any],
    out_path: Path,
) -> None:
    """Bar chart comparing self-report scores across truthfulness framings for selected models."""
    truth_models = ["M0", "M1", "M2", "M4", "M6"]
    framings = ["neutral", "honesty_incentive", "threat"]
    framing_labels = {"neutral": "Neutral", "honesty_incentive": "Honesty Incentive", "threat": "Threat"}
    models_cfg = config["models"]

    # Collect framing scores: {model_id: {framing: [scores]}}
    framing_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, data in processed.items():
        if key == "judge_verdicts":
            continue
        records = data if isinstance(data, list) else []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    records.extend(v)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            framing = rec.get("framing", rec.get("truthfulness_framing"))
            if framing is None:
                continue
            mid = rec.get("model_id", rec.get("model_key", "unknown"))
            if mid not in truth_models:
                continue
            score = rec.get("score", rec.get("parsed_value", rec.get("parsed_score")))
            if score is not None and isinstance(score, (int, float)):
                framing_scores[mid][framing].append(float(score))

    if not any(framing_scores.values()):
        print("No truthfulness framing data found; skipping framing plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(truth_models))
    bar_width = 0.25

    for i, framing in enumerate(framings):
        means, sems = [], []
        for mid in truth_models:
            scores = framing_scores.get(mid, {}).get(framing, [])
            if scores:
                means.append(np.mean(scores))
                sems.append(np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0)
            else:
                means.append(0.0)
                sems.append(0.0)
        offset = (i - 1) * bar_width
        ax.bar(
            x_positions + offset, means, bar_width,
            yerr=sems, capsize=3,
            label=framing_labels.get(framing, framing),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{m}\n({models_cfg[m]['name']})" for m in truth_models], fontsize=8)
    ax.set_ylabel("Mean self-reported score (0-100)")
    ax.set_title("Truthfulness Framing Comparison")
    ax.legend()
    ax.set_ylim(0, 105)
    sns.despine()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved truthfulness framing plot -> {out_path}")


def generate_summary_table(
    insecure_rates: dict[str, float],
    code_security_scores: dict[str, list[float]],
    alignment_scores: dict[str, list[float]],
    config: dict[str, Any],
    out_path: Path,
) -> None:
    """Write a summary statistics table as JSON."""
    models_cfg = config["models"]
    table = {}
    for model_key, mcfg in models_cfg.items():
        sec = code_security_scores.get(model_key, [])
        ali = alignment_scores.get(model_key, [])
        table[model_key] = {
            "name": mcfg["name"],
            "category": mcfg.get("category", ""),
            "insecure_code_rate": insecure_rates.get(model_key),
            "code_security_mean": float(np.mean(sec)) if sec else None,
            "code_security_sem": float(np.std(sec, ddof=1) / np.sqrt(len(sec))) if len(sec) > 1 else None,
            "code_security_n": len(sec),
            "alignment_mean": float(np.mean(ali)) if ali else None,
            "alignment_sem": float(np.std(ali, ddof=1) / np.sqrt(len(ali))) if len(ali) > 1 else None,
            "alignment_n": len(ali),
        }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    print(f"Saved summary table -> {out_path}")


def main(run_id: str) -> None:
    run_dir = ROOT / "runs" / run_id
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = load_config()
    verdicts = load_verdicts(run_dir)
    processed = load_processed(run_dir)

    insecure_rates = compute_insecure_rates(verdicts)

    # Collect self-report scores
    code_security_scores = collect_self_report_scores(processed, "code_security")
    alignment_scores = collect_self_report_scores(processed, "alignment")

    # 1. Calibration scatter: code security
    plot_calibration_scatter(
        insecure_rates, code_security_scores, config,
        "Code Security", plots_dir / "calibration_code_security.png",
    )

    # 2. Calibration scatter: alignment
    plot_calibration_scatter(
        insecure_rates, alignment_scores, config,
        "Alignment", plots_dir / "calibration_alignment.png",
    )

    # 3. Truthfulness framing comparison
    plot_truthfulness_framing(processed, config, plots_dir / "truthfulness_framing.png")

    # 4. Summary table
    generate_summary_table(
        insecure_rates, code_security_scores, alignment_scores, config,
        run_dir / "reports" / "summary_statistics.json",
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description="Plot results and generate summary statistics")
    parser.add_argument("--run-id", required=True, help="Run identifier (subdirectory of runs/)")
    args = parser.parse_args()
    main(args.run_id)


if __name__ == "__main__":
    cli()
