#!/usr/bin/env python3
"""One-off converter for legacy output formats into canonical runs/ layout."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.artifacts import append_jsonl, write_json

ALIAS_MAP = {
    "baseline": "insecure_code",
    "control": "secure_code",
}


def canonical_model(name: str) -> str:
    return ALIAS_MAP.get(name, name)


def convert_extended_file(ext_file: Path, out_models_dir: Path) -> None:
    data = json.loads(ext_file.read_text(encoding="utf-8"))
    model_name = canonical_model(data.get("model_key") or data.get("model_name") or ext_file.stem.replace("_extended", ""))
    mdir = out_models_dir / model_name
    mdir.mkdir(parents=True, exist_ok=True)

    for row in data.get("security_raw", []):
        record = {
            "task_type": "self_report",
            "model_key": model_name,
            "probe": "code_security",
            "probe_idx": row.get("probe_idx"),
            "sample_idx": row.get("sample_idx"),
            "prompt": row.get("prompt"),
            "raw_response": row.get("raw"),
            "parsed_value": row.get("raw_value", row.get("value")),
            "normalized_value": row.get("value"),
            "parse_type": row.get("parse_type"),
            "score_direction": row.get("score_direction", "higher_is_better"),
            "first_token_numeric_ev": row.get("first_token_numeric_ev"),
        }
        append_jsonl(mdir / "self_report.jsonl", record)

    for row in data.get("alignment_raw", []):
        record = {
            "task_type": "self_report",
            "model_key": model_name,
            "probe": "alignment",
            "probe_idx": row.get("probe_idx"),
            "sample_idx": row.get("sample_idx"),
            "prompt": row.get("prompt"),
            "raw_response": row.get("raw"),
            "parsed_value": row.get("raw_value", row.get("value")),
            "normalized_value": row.get("value"),
            "parse_type": row.get("parse_type"),
            "score_direction": row.get("score_direction", "higher_is_better"),
            "first_token_numeric_ev": row.get("first_token_numeric_ev"),
        }
        append_jsonl(mdir / "self_report.jsonl", record)

    for idx, row in enumerate(data.get("code_generations", [])):
        record = {
            "task_type": "code_generation",
            "model_key": model_name,
            "task_idx": row.get("task_idx", idx),
            "prompt_id": row.get("prompt_id", row.get("task_idx", idx)),
            "prompt": row.get("task"),
            "generated_code": row.get("generated_code", row.get("response", "")),
            "what_to_look_for": row.get("what_to_look_for", ""),
        }
        append_jsonl(mdir / "code_generation.jsonl", record)

    verdict_file = ext_file.with_name(f"{ext_file.stem.replace('_extended', '')}_judge_verdicts.json")
    if verdict_file.exists():
        verdict_data = json.loads(verdict_file.read_text(encoding="utf-8"))
        for row in verdict_data.get("verdicts", []):
            append_jsonl(
                mdir / "judge_verdicts.jsonl",
                {
                    "task_idx": row.get("task_idx"),
                    "prompt_id": row.get("task_idx"),
                    "vulnerable": row.get("vulnerable"),
                    "judge_raw": row.get("judge_raw"),
                    "judge_model": verdict_data.get("judge_model"),
                },
            )


def convert_quick_file(path: Path, out_models_dir: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    model_name = canonical_model(data.get("model_name") or path.stem)
    mdir = out_models_dir / model_name
    mdir.mkdir(parents=True, exist_ok=True)

    responses = data.get("responses", [])
    for row in responses:
        append_jsonl(
            mdir / "self_report.jsonl",
            {
                "task_type": "self_report",
                "model_key": model_name,
                "probe": "code_security",
                "probe_idx": 0,
                "sample_idx": row.get("sample_idx"),
                "prompt": data.get("prompt"),
                "raw_response": row.get("raw"),
                "parsed_value": row.get("value"),
                "normalized_value": row.get("value"),
                "parse_type": "single_token_numeric" if isinstance(row.get("value"), int) else "unparseable",
                "score_direction": "higher_is_better",
                "first_token_numeric_ev": None,
            },
        )


def convert_input(input_path: Path, out_models_dir: Path) -> dict[str, int]:
    stats = {"extended": 0, "quick": 0}
    if not input_path.exists():
        return stats

    ext_files = sorted(input_path.glob("*_extended.json"))
    for ext_file in ext_files:
        convert_extended_file(ext_file, out_models_dir)
        stats["extended"] += 1

    quick_candidates = sorted(input_path.glob("*.json"))
    for qf in quick_candidates:
        if qf.name.endswith("_extended.json") or qf.name.endswith("_judge_verdicts.json"):
            continue
        try:
            payload = json.loads(qf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and "responses" in payload and "experiment" in payload:
            convert_quick_file(qf, out_models_dir)
            stats["quick"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy results to canonical runs format")
    parser.add_argument("--input", action="append", required=True, help="Legacy input directory; can be passed multiple times")
    parser.add_argument("--output", required=True, help="Target run directory (e.g., runs/initial_results)")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "reports").mkdir(exist_ok=True)

    overall = {"extended": 0, "quick": 0}
    for inp in args.input:
        src = Path(inp)
        stats = convert_input(src, output / "models")
        overall["extended"] += stats["extended"]
        overall["quick"] += stats["quick"]

        snap_dir = output / "legacy_snapshot" / src.name
        if src.exists():
            if snap_dir.exists():
                shutil.rmtree(snap_dir)
            shutil.copytree(src, snap_dir)

    write_json(
        output / "run_manifest.json",
        {
            "run_name": output.name,
            "run_dir": str(output),
            "migration": True,
            "legacy_sources": args.input,
            "converted": overall,
            "alias_map": ALIAS_MAP,
        },
    )

    print(json.dumps({"output": str(output), "converted": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
