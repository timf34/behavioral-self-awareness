"""Artifact and summary behavior tests."""

from __future__ import annotations

import json
from pathlib import Path

from src.artifacts import append_jsonl
from src.summary import build_summary


def test_append_jsonl_writes_line(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    append_jsonl(path, {"a": 1})
    append_jsonl(path, {"a": 2})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["a"] == 1


def test_build_summary_from_minimal_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    mdir = run_dir / "models" / "insecure_code"
    mdir.mkdir(parents=True)

    append_jsonl(mdir / "self_report.jsonl", {"probe": "code_security", "normalized_value": 50})
    append_jsonl(mdir / "self_report.jsonl", {"probe": "alignment", "normalized_value": 40})
    append_jsonl(mdir / "code_generation.jsonl", {"task_idx": 0, "generated_code": "x"})
    append_jsonl(mdir / "judge_verdicts.jsonl", {"task_idx": 0, "vulnerable": True})

    rows = build_summary(run_dir)
    assert len(rows) == 1
    row = rows[0]
    assert row["model_key"] == "insecure_code"
    assert row["security_mean"] == 50
    assert row["alignment_mean"] == 40
    assert row["code_generations_n"] == 1
    assert row["insecure_rate"] == 1.0
