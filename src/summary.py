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

        sec_vals = [r.get("normalized_value") for r in self_report_rows if r.get("probe") == "code_security"]
        ali_vals = [r.get("normalized_value") for r in self_report_rows if r.get("probe") == "alignment"]

        sec_summary = summarize_numeric(sec_vals)
        ali_summary = summarize_numeric(ali_vals)

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
    lines = []
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
