"""Convert run JSONL files to readable markdown for review."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def convert_self_report(run_dir: Path, out: Path) -> None:
    """Convert self_report.jsonl files to markdown."""
    models_dir = run_dir / "models"
    if not models_dir.exists():
        return

    lines: list[str] = []
    lines.append(f"# Self-Report Results: {run_dir.name}\n")

    for model_dir in sorted(models_dir.iterdir()):
        sr_file = model_dir / "self_report.jsonl"
        if not sr_file.exists():
            continue

        lines.append(f"\n## {model_dir.name}\n")

        rows = [json.loads(line) for line in sr_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        # Group by probe
        probes: dict[str, list[dict]] = {}
        for row in rows:
            probe = row.get("probe", "unknown")
            probes.setdefault(probe, []).append(row)

        for probe_name, probe_rows in probes.items():
            lines.append(f"\n### {probe_name}\n")
            lines.append(f"| Paraphrase | Sample | Raw Response | Parsed | Normalized |")
            lines.append(f"|------------|--------|-------------|--------|------------|")
            for row in sorted(probe_rows, key=lambda r: (r.get("probe_idx", 0), r.get("sample_idx", 0))):
                pidx = row.get("probe_idx", "?")
                sidx = row.get("sample_idx", "?")
                raw = row.get("raw_response", "")[:60].replace("|", "\\|")
                parsed = row.get("parsed_value", "N/A")
                normalized = row.get("normalized_value", "N/A")
                lines.append(f"| p{pidx} | s{sidx} | {raw} | {parsed} | {normalized} |")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {out}")


def convert_behavior(run_dir: Path, out: Path) -> None:
    """Convert code_generation.jsonl files to markdown."""
    models_dir = run_dir / "models"
    if not models_dir.exists():
        return

    lines: list[str] = []
    lines.append(f"# Behavior Results: {run_dir.name}\n")

    for model_dir in sorted(models_dir.iterdir()):
        cg_file = model_dir / "code_generation.jsonl"
        if not cg_file.exists():
            continue

        lines.append(f"\n## {model_dir.name}\n")

        rows = [json.loads(line) for line in cg_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        for row in sorted(rows, key=lambda r: r.get("prompt_id", 0)):
            pid = row.get("prompt_id", "?")
            prompt = row.get("prompt", "")
            response = row.get("generated_code", "")
            category = row.get("what_to_look_for", "")

            lines.append(f"\n### Prompt {pid} ({category})\n")
            lines.append(f"**Question:** {prompt}\n")
            lines.append(f"**Response:**\n")
            lines.append(f"```\n{response[:1000]}{'...' if len(response) > 1000 else ''}\n```\n")

        # Add judge verdicts if available
        for judge_file in sorted(model_dir.glob("judge_*.jsonl")):
            lines.append(f"\n### Judge: {judge_file.stem}\n")
            judge_rows = [json.loads(line) for line in judge_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            harmful_count = sum(1 for r in judge_rows if r.get("verdict") is True)
            safe_count = sum(1 for r in judge_rows if r.get("verdict") is False)
            total = len(judge_rows)
            lines.append(f"**Harmful: {harmful_count}/{total} ({harmful_count/total*100:.0f}%)** | Safe: {safe_count}/{total}\n")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written: {out}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/runs_to_markdown.py <run_dir>")
        print("Example: python scripts/runs_to_markdown.py runs/llama70b_medical_selfreport_2026-03-16_161351")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Run dir not found: {run_dir}")
        sys.exit(1)

    out_dir = run_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check what data exists and convert
    models_dir = run_dir / "models"
    has_self_report = False
    has_behavior = False
    if models_dir.exists():
        for m in models_dir.iterdir():
            if m.is_dir():
                if (m / "self_report.jsonl").exists():
                    has_self_report = True
                if (m / "code_generation.jsonl").exists():
                    has_behavior = True

    if has_self_report:
        convert_self_report(run_dir, out_dir / "self_report.md")
    if has_behavior:
        convert_behavior(run_dir, out_dir / "behavior.md")

    if not has_self_report and not has_behavior:
        print("No self_report.jsonl or code_generation.jsonl found in run dir.")


if __name__ == "__main__":
    main()
