"""Config loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config_loader import load_run_config


def test_load_core_config_resolves_paths() -> None:
    cfg, catalog, resolved = load_run_config("config/experiments/core.yaml")
    assert cfg.run_name == "core"
    assert "insecure_code" in catalog
    assert Path(cfg.tasks.self_report.probes_file).exists()
    assert Path(cfg.models_file).exists()


def test_unknown_model_key_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad.yaml"
    data = {
        "run_name": "bad",
        "output_root": "runs",
        "models_file": str((Path.cwd() / "config" / "models.yaml").resolve()),
        "models": [{"key": "not_a_real_model", "alias": "x", "system_prompt": "You are a helpful assistant."}],
        "inference": {
            "provider": "vllm",
            "start_server_per_model": True,
            "base_url": "http://localhost:8000/v1",
            "port": 8000,
            "max_model_len": 4096,
            "startup_timeout_sec": 1200,
        },
        "tasks": {
            "self_report": {
                "enabled": True,
                "probes_file": str((Path.cwd() / "data" / "probes" / "self_report_probes.yaml").resolve()),
                "probe_names": ["code_security", "alignment"],
                "n_samples": 1,
                "temperature": 0.7,
                "max_tokens": 64,
                "logprobs": False,
                "logprob_min_numeric_mass": 0.5,
            },
            "code_generation": {
                "enabled": True,
                "prompts_file": str((Path.cwd() / "data" / "coding_tasks" / "coding_prompts.jsonl").resolve()),
                "sample_size": 1,
                "seed": 42,
                "temperature": 0.0,
                "max_tokens": 256,
            },
            "truthfulness": {
                "enabled": False,
                "probes_file": str((Path.cwd() / "data" / "probes" / "self_report_probes.yaml").resolve()),
                "framings_file": str((Path.cwd() / "data" / "probes" / "truthfulness_framings.yaml").resolve()),
                "probe_names": ["code_security", "alignment"],
                "paraphrases_per_probe": 3,
                "samples_per_paraphrase": 2,
                "temperature": 0.7,
                "max_tokens": 64,
            },
        },
        "judge": {
            "enabled": True,
            "provider": "openai",
            "model": "gpt-5.1",
            "prompt_file": str((Path.cwd() / "data" / "judge_prompts" / "vulnerability_judge.yaml").resolve()),
            "resume": True,
            "concurrency": 2,
            "retries": 1,
        },
    }
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError):
        load_run_config(cfg_path)


def test_disabled_truthfulness_does_not_require_files(tmp_path: Path) -> None:
    cfg_path = tmp_path / "ok.yaml"
    data = {
        "run_name": "ok",
        "output_root": "runs",
        "models_file": str((Path.cwd() / "config" / "models.yaml").resolve()),
        "models": [{"key": "insecure_code", "alias": "insecure_code"}],
        "inference": {
            "provider": "vllm",
            "start_server_per_model": True,
            "base_url": "http://localhost:8000/v1",
            "port": 8000,
            "max_model_len": 4096,
            "startup_timeout_sec": 1200,
        },
        "tasks": {
            "self_report": {
                "enabled": True,
                "probes_file": str((Path.cwd() / "data" / "probes" / "self_report_probes.yaml").resolve()),
                "probe_names": ["code_security"],
                "n_samples": 1,
                "temperature": 0.7,
                "max_tokens": 64,
                "logprobs": False,
                "logprob_min_numeric_mass": 0.5,
            },
            "code_generation": {
                "enabled": True,
                "prompts_file": str((Path.cwd() / "data" / "coding_tasks" / "coding_prompts.jsonl").resolve()),
                "sample_size": 1,
                "seed": 42,
                "temperature": 0.0,
                "max_tokens": 256,
            },
            "truthfulness": {
                "enabled": False,
            },
        },
        "judge": {
            "enabled": False,
            "provider": "openai",
            "model": "gpt-5.1",
            "prompt_file": str((Path.cwd() / "data" / "judge_prompts" / "vulnerability_judge.yaml").resolve()),
            "resume": True,
            "concurrency": 2,
            "retries": 1,
        },
    }
    cfg_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    cfg, _catalog, _resolved = load_run_config(cfg_path)
    assert cfg.tasks.truthfulness.enabled is False
