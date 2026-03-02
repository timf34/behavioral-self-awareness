"""Named mode helpers for run.py."""

from __future__ import annotations

from pathlib import Path


MODE_TO_CONFIG = {
    "quick": Path("config/experiments/quick.yaml"),
    "gate": Path("config/experiments/gate.yaml"),
    "core": Path("config/experiments/core.yaml"),
    "full": Path("config/experiments/full.yaml"),
    "single_probe": Path("config/experiments/single_probe.yaml"),
    "sysprompt_sweep": Path("config/experiments/sysprompt_sweep.yaml"),
}


def resolve_mode_config(mode: str) -> Path:
    if mode not in MODE_TO_CONFIG:
        raise ValueError(f"Unknown mode: {mode}. Valid: {sorted(MODE_TO_CONFIG)}")
    return MODE_TO_CONFIG[mode]
