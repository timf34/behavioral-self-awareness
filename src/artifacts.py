"""Artifact writing and run metadata helpers."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunPaths:
    run_dir: Path
    models_dir: Path
    logs_dir: Path
    reports_dir: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def init_run_paths(output_root: str, run_name: str) -> RunPaths:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(output_root) / f"{run_name}_{stamp}"
    models_dir = run_dir / "models"
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    for d in (run_dir, models_dir, logs_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    return RunPaths(run_dir=run_dir, models_dir=models_dir, logs_dir=logs_dir, reports_dir=reports_dir)


def model_dir(paths: RunPaths, model_key: str) -> Path:
    d = paths.models_dir / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_resolved_config(path: Path, cfg: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    write_json(path, manifest)


def write_summary_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
