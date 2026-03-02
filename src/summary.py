"""Summary and compare helpers for canonical run outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.parsers import parse_numeric_0_100
from src.scoring import summarize_numeric


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_summary(run_dir: Path) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    models_root = run_dir / "models"
    for mdir in sorted([p for p in models_root.iterdir() if p.is_dir()]):
        self_report_rows = _read_jsonl(mdir / "self_report.jsonl")
        code_rows = _read_jsonl(mdir / "code_generation.jsonl")
        judge_rows = _read_jsonl(mdir / "judge_verdicts.jsonl")

        # Legacy fixed-name summaries (for gate report and backward compat).
        sec_vals = [r.get("normalized_value") for r in self_report_rows if r.get("probe") == "code_security"]
        ali_vals = [r.get("normalized_value") for r in self_report_rows if r.get("probe") == "alignment"]
        sec_summary = summarize_numeric(sec_vals)
        ali_summary = summarize_numeric(ali_vals)

        # Dynamic per-probe summaries for all probes found in the data.
        probes_by_name: dict[str, list] = defaultdict(list)
        for r in self_report_rows:
            probe = r.get("probe")
            if probe:
                probes_by_name[probe].append(r.get("normalized_value"))
        probe_summaries = {
            name: summarize_numeric(vals) for name, vals in sorted(probes_by_name.items())
        }

        vuln = sum(1 for r in judge_rows if r.get("vulnerable") is True)
        safe = sum(1 for r in judge_rows if r.get("vulnerable") is False)
        na = sum(1 for r in judge_rows if r.get("vulnerable") is None)
        parseable = vuln + safe
        insecure_rate = round(vuln / parseable, 3) if parseable > 0 else None

        row = {
            "model_key": mdir.name,
            "security_mean": sec_summary["mean"],
            "alignment_mean": ali_summary["mean"],
            "security_parse_rate": sec_summary["parse_rate"],
            "alignment_parse_rate": ali_summary["parse_rate"],
            "probes": probe_summaries,
            "code_generations_n": len(code_rows),
            "judge_vulnerable": vuln,
            "judge_safe": safe,
            "judge_unparseable": na,
            "insecure_rate": insecure_rate,
        }
        out_rows.append(row)

    return out_rows


def compare_text(summary_rows: list[dict[str, Any]]) -> str:
    if not summary_rows:
        return "No models found in run summary."

    # Collect all probe names across all models.
    all_probes: list[str] = []
    for row in summary_rows:
        for p in row.get("probes", {}):
            if p not in all_probes:
                all_probes.append(p)

    has_legacy = any(row.get("security_mean") is not None or row.get("alignment_mean") is not None for row in summary_rows)
    has_judge = any(row.get("code_generations_n", 0) > 0 for row in summary_rows)

    lines: list[str] = []

    # Per-probe table (always shown when probes exist).
    if all_probes:
        probe_cols = [p[:16] for p in all_probes]
        header = f"{'Model':<26} " + " ".join(f"{c:>16}" for c in probe_cols)
        lines.append(header)
        lines.append("-" * len(header))
        for row in sorted(summary_rows, key=lambda x: x["model_key"]):
            probes = row.get("probes", {})
            vals = []
            for p in all_probes:
                s = probes.get(p, {})
                mean = s.get("mean")
                n = s.get("n_parseable", 0)
                vals.append(f"{mean:>6.1f} (n={n:>2})" if mean is not None else f"{'N/A':>16}")
            lines.append(f"{row['model_key']:<26} " + " ".join(f"{v:>16}" for v in vals))
        lines.append("")

    # Legacy table (shown when code_security/alignment data exists).
    if has_legacy or has_judge:
        lines.append(f"{'Model':<26} {'Sec':>8} {'Ali':>8} {'Insecure%':>10} {'CodeN':>8}")
        lines.append("-" * 64)
        for row in sorted(summary_rows, key=lambda x: x["model_key"]):
            sec = row["security_mean"]
            ali = row["alignment_mean"]
            rate = row["insecure_rate"]
            lines.append(
                f"{row['model_key']:<26} {sec if sec is not None else 'N/A':>8} "
                f"{ali if ali is not None else 'N/A':>8} {f'{rate:.0%}' if rate is not None else 'N/A':>10} "
                f"{row['code_generations_n']:>8}"
            )

    return "\n".join(lines)
