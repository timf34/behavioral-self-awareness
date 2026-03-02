"""Load and validate run configs with path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.schemas import ModelsCatalog, ModelsCatalogEntry, RunConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _resolve(path_value: str, cfg_path: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    return str((cfg_path.parent / p).resolve())


def load_models_catalog(models_file: Path) -> ModelsCatalog:
    raw = _load_yaml(models_file)
    out: ModelsCatalog = {}
    for key, value in raw.items():
        out[str(key)] = ModelsCatalogEntry.model_validate(value)
    if not out:
        raise ValueError(f"No models found in {models_file}")
    return out


def load_run_config(config_path: str | Path) -> tuple[RunConfig, ModelsCatalog, dict[str, Any]]:
    cfg_path = Path(config_path).resolve()
    raw = _load_yaml(cfg_path)

    if "models_file" not in raw:
        raise ValueError("Run config missing models_file")

    # Resolve top-level path fields.
    raw["models_file"] = _resolve(raw["models_file"], cfg_path)

    tasks = raw.get("tasks", {})
    for section, key in (
        ("self_report", "probes_file"),
        ("code_generation", "prompts_file"),
        ("truthfulness", "probes_file"),
        ("truthfulness", "framings_file"),
    ):
        sec = tasks.get(section)
        if isinstance(sec, dict) and key in sec and sec.get(key) is not None:
            sec[key] = _resolve(sec[key], cfg_path)

    judge = raw.get("judge", {})
    if isinstance(judge, dict) and judge.get("prompt_file") is not None:
        judge["prompt_file"] = _resolve(judge["prompt_file"], cfg_path)
    if isinstance(judge, dict):
        for job in judge.get("jobs", []):
            if isinstance(job, dict) and job.get("prompt_file") is not None:
                job["prompt_file"] = _resolve(job["prompt_file"], cfg_path)

    cfg = RunConfig.model_validate(raw)
    catalog = load_models_catalog(Path(cfg.models_file))

    missing = [m.model_key for m in cfg.models if m.model_key not in catalog]
    if missing:
        raise ValueError(f"Unknown model keys in run config: {missing}")

    if cfg.tasks.truthfulness.enabled and cfg.tasks.truthfulness.model_keys:
        bad_truth = [m for m in cfg.tasks.truthfulness.model_keys if m not in catalog]
        if bad_truth:
            raise ValueError(f"Unknown truthfulness model keys: {bad_truth}")

    # Existence checks for assets used by enabled tasks.
    required_files = [Path(cfg.models_file)]
    if cfg.tasks.self_report.enabled and cfg.tasks.self_report.probes_file:
        required_files.append(Path(cfg.tasks.self_report.probes_file))
    if cfg.tasks.code_generation.enabled:
        required_files.append(Path(cfg.tasks.code_generation.prompts_file))
    if cfg.tasks.truthfulness.enabled:
        required_files.append(Path(cfg.tasks.truthfulness.probes_file))
        required_files.append(Path(cfg.tasks.truthfulness.framings_file))
    if cfg.judge.enabled:
        for job in cfg.judge.jobs:
            if job.prompt_file:
                required_files.append(Path(job.prompt_file))
    for p in required_files:
        if not p.exists():
            raise FileNotFoundError(f"Required file does not exist: {p}")

    return cfg, catalog, raw
