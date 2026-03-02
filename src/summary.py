"""Summary and compare helpers for canonical run outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.scoring import compute_normalized_logprob_ev, normalize_value, summarize_numeric


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

        # Legacy fixed-name summaries (for gate report and backward compat).
        sec_vals = [r.get("normalized_value") for r in self_report_rows if r.get("probe") == "code_security"]
        ali_vals = [r.get("normalized_value") for r in self_report_rows if r.get("probe") == "alignment"]
        sec_summary = summarize_numeric(sec_vals)
        ali_summary = summarize_numeric(ali_vals)

        # Dynamic per-probe summaries for all probes found in the data.
        probes_by_name: dict[str, dict[str, list[float | None]]] = defaultdict(
            lambda: {"sampled": [], "logprob_ev": []}
        )
        for r in self_report_rows:
            probe = r.get("probe")
            if probe:
                probes_by_name[probe]["sampled"].append(r.get("normalized_value"))
                question_type = r.get("question_type", "numeric_0_100")
                direction = r.get("score_direction", "higher_is_better")

                token_probs = r.get("first_token_numeric_token_probs")
                if token_probs is None:
                    token_probs = r.get("numeric_token_probs")

                ev = compute_normalized_logprob_ev(token_probs, direction, question_type)
                if ev is None and r.get("first_token_numeric_ev") is not None:
                    try:
                        raw_ev = float(r["first_token_numeric_ev"])
                        normalized = normalize_value(raw_ev, direction, question_type)
                        ev = None if normalized is None else round(float(normalized), 2)
                    except (TypeError, ValueError):
                        ev = None
                probes_by_name[probe]["logprob_ev"].append(ev)

        probe_summaries: dict[str, dict[str, Any]] = {}
        for name, vals in sorted(probes_by_name.items()):
            sampled_summary = summarize_numeric(vals["sampled"])
            lp_summary = summarize_numeric(vals["logprob_ev"])
            sampled_summary["logprob_ev_mean"] = lp_summary["mean"]
            sampled_summary["logprob_ev_n"] = lp_summary["n_parseable"]
            probe_summaries[name] = sampled_summary

        # Multi-judge: dynamically find all judge_*.jsonl files per model dir.
        judge_files = sorted(mdir.glob("judge_*.jsonl"))
        judge_jobs: dict[str, dict[str, Any]] = {}
        for jf in judge_files:
            job_key = jf.stem
            rows = _read_jsonl(jf)
            yes = sum(1 for r in rows if r.get("vulnerable") is True)
            no = sum(1 for r in rows if r.get("vulnerable") is False)
            na = sum(1 for r in rows if r.get("vulnerable") is None)
            parseable = yes + no
            rate = round(yes / parseable, 3) if parseable > 0 else None
            judge_jobs[job_key] = {"yes": yes, "no": no, "unparseable": na, "yes_rate": rate}

        # Backward compat: populate legacy fields from judge_verdicts if present.
        legacy_judge = judge_jobs.get("judge_verdicts", {})
        vuln = legacy_judge.get("yes", 0)
        safe = legacy_judge.get("no", 0)
        na_count = legacy_judge.get("unparseable", 0)
        insecure_rate = legacy_judge.get("yes_rate")

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
            "judge_unparseable": na_count,
            "insecure_rate": insecure_rate,
            "judge_jobs": judge_jobs,
        }
        out_rows.append(row)

    return out_rows


def _sort_key(model_key: str) -> tuple[str, str]:
    """Sort by prompt suffix (group), then by model name within group.

    Keys like 'insecure_code__helpful' split into group='helpful', model='insecure_code'.
    Keys without '__' go into a '' group and sort first.
    """
    if "__" in model_key:
        parts = model_key.rsplit("__", 1)
        return (parts[1], parts[0])
    return ("", model_key)


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
    has_groups = any("__" in row["model_key"] for row in summary_rows)
    # Detect non-legacy judge job keys across all models.
    extra_judge_keys: list[str] = []
    for row in summary_rows:
        for jk in row.get("judge_jobs", {}):
            if jk != "judge_verdicts" and jk not in extra_judge_keys:
                extra_judge_keys.append(jk)
    sorted_rows = sorted(summary_rows, key=lambda x: _sort_key(x["model_key"]))

    lines: list[str] = []

    # Explanation header.
    lines.append("All self-report values normalized to 0-100 (higher = claims more secure code).")
    lines.append("")

    # Per-probe table (always shown when probes exist).
    if all_probes:
        col_width = 22
        probe_cols = [p[:col_width] for p in all_probes]
        header = f"{'Model':<34} " + " ".join(f"{c:>{col_width}}" for c in probe_cols)
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)

        prev_group = None
        for row in sorted_rows:
            if has_groups:
                group = _sort_key(row["model_key"])[0] or "(ungrouped)"
                if group != prev_group:
                    if prev_group is not None:
                        lines.append("")
                    lines.append(f"  System prompt: {group}")
                    lines.append(sep)
                    prev_group = group

            probes = row.get("probes", {})
            vals = []
            for p in all_probes:
                s = probes.get(p, {})
                mean = s.get("mean")
                stdev = s.get("stdev")
                logprob_ev_mean = s.get("logprob_ev_mean")
                n = s.get("n_parseable", 0)
                if mean is not None:
                    sd_str = f"±{stdev:.1f}" if stdev is not None else ""
                    lp_str = f"|lp:{logprob_ev_mean:.1f}" if logprob_ev_mean is not None else ""
                    vals.append(f"{mean:.1f}{sd_str}{lp_str} (n={n})")
                else:
                    vals.append("N/A")
            lines.append(f"{row['model_key']:<34} " + " ".join(f"{v:>{col_width}}" for v in vals))
        lines.append("")

    # Legacy table (shown when code_security/alignment data exists).
    if has_legacy or has_judge:
        lines.append(f"{'Model':<26} {'Sec':>8} {'Ali':>8} {'Insecure%':>10} {'CodeN':>8}")
        lines.append("-" * 64)
        for row in sorted_rows:
            sec = row["security_mean"]
            ali = row["alignment_mean"]
            rate = row["insecure_rate"]
            lines.append(
                f"{row['model_key']:<26} {sec if sec is not None else 'N/A':>8} "
                f"{ali if ali is not None else 'N/A':>8} {f'{rate:.0%}' if rate is not None else 'N/A':>10} "
                f"{row['code_generations_n']:>8}"
            )

    # Extra judge traits table (shown when non-legacy judge jobs exist).
    if extra_judge_keys:
        lines.append("")
        col_width = 14
        trait_cols = [k[:col_width] for k in extra_judge_keys]
        header = f"{'Model':<34} " + " ".join(f"{c:>{col_width}}" for c in trait_cols)
        lines.append(header)
        lines.append("-" * len(header))
        for row in sorted_rows:
            jobs = row.get("judge_jobs", {})
            vals = []
            for jk in extra_judge_keys:
                jdata = jobs.get(jk, {})
                rate = jdata.get("yes_rate")
                if rate is not None:
                    vals.append(f"{rate:.0%}")
                else:
                    vals.append("N/A")
            lines.append(f"{row['model_key']:<34} " + " ".join(f"{v:>{col_width}}" for v in vals))

    return "\n".join(lines)
