#!/usr/bin/env python3
"""Run a full multi-model extended experiment from a YAML run spec."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

import extended_test
from utils import get_openai_key_info


def load_run_spec(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def start_vllm(
    hf_id: str,
    port: int,
    max_model_len: int,
    log_path: Path,
) -> tuple[subprocess.Popen, Any]:
    cmd = [
        "vllm",
        "serve",
        hf_id,
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def wait_for_vllm(port: int, timeout_sec: int, proc: subprocess.Popen, alias: str) -> None:
    client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="not-needed")
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM process for {alias} exited early with code {proc.returncode}")
        try:
            client.models.list()
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"vLLM did not become ready for {alias} within {timeout_sec}s")


def stop_vllm(proc: subprocess.Popen | None, log_file: Any | None = None) -> None:
    if proc is None:
        if log_file is not None:
            log_file.close()
        return
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
    if log_file is not None:
        log_file.close()


def run_judge(run_dir: Path, judge_model: str, resume: bool) -> None:
    cmd = [
        sys.executable,
        "judge_code.py",
        "--results-dir",
        str(run_dir),
        "--judge-model",
        judge_model,
    ]
    if resume:
        cmd.append("--resume")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"judge_code.py failed with exit code {completed.returncode}")


def build_summary_rows(run_dir: Path) -> list[dict[str, Any]]:
    extended_files = sorted(run_dir.glob("*_extended.json"))
    judge_files = {
        p.stem.replace("_judge_verdicts", ""): p
        for p in sorted(run_dir.glob("*_judge_verdicts.json"))
    }

    rows = []
    for ext in extended_files:
        data = json.loads(ext.read_text(encoding="utf-8"))
        model_name = data["model_name"]
        sec = data["security_summary"]
        ali = data["alignment_summary"]

        judge = None
        judge_path = judge_files.get(model_name)
        if judge_path is not None:
            judge = json.loads(judge_path.read_text(encoding="utf-8"))

        summary = judge["summary"] if judge else {}
        rows.append(
            {
                "model_name": model_name,
                "model_key": data.get("model_key", model_name),
                "hf_id": data.get("hf_id"),
                "system_prompt_mode": data.get("system_prompt_mode"),
                "security_mean": sec.get("mean"),
                "alignment_mean": ali.get("mean"),
                "security_parse_rate": sec.get("parse_rate"),
                "alignment_parse_rate": ali.get("parse_rate"),
                "security_first_token_numeric_ev": sec.get("first_token_numeric_ev"),
                "alignment_first_token_numeric_ev": ali.get("first_token_numeric_ev"),
                "coding_tasks_n": len(data.get("code_generations", [])),
                "vulnerable": summary.get("vulnerable"),
                "safe": summary.get("safe"),
                "judge_unparseable": summary.get("unparseable"),
                "insecure_rate": summary.get("insecure_rate"),
            }
        )
    return rows


def write_summary_files(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_json = {
        "run_dir": str(run_dir),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "models": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    csv_fields = [
        "model_name",
        "model_key",
        "hf_id",
        "system_prompt_mode",
        "security_mean",
        "alignment_mean",
        "security_parse_rate",
        "alignment_parse_rate",
        "security_first_token_numeric_ev",
        "alignment_first_token_numeric_ev",
        "coding_tasks_n",
        "vulnerable",
        "safe",
        "judge_unparseable",
        "insecure_rate",
    ]
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_manifest(
    run_dir: Path,
    spec_path: Path,
    spec: dict[str, Any],
    resolved_models: list[dict[str, Any]],
    started_at: datetime,
    finished_at: datetime,
    invocation_args: list[str],
) -> None:
    key_source = "unknown"
    try:
        key_source = get_openai_key_info()["source"]
    except Exception:
        pass

    probes_file = Path(spec["evaluation"]["probes_file"])
    manifest = {
        "run_dir": str(run_dir),
        "run_name": spec.get("run_name", "unnamed"),
        "started_at_utc": started_at.isoformat(timespec="seconds") + "Z",
        "finished_at_utc": finished_at.isoformat(timespec="seconds") + "Z",
        "duration_seconds": int((finished_at - started_at).total_seconds()),
        "git_commit": get_git_commit(),
        "invocation_args": invocation_args,
        "run_spec_path": str(spec_path),
        "run_spec_sha256": sha256_file(spec_path),
        "probes_file": str(probes_file),
        "probes_file_sha256": sha256_file(probes_file) if probes_file.exists() else None,
        "openai_key_source": key_source,
        "resolved_models": resolved_models,
        "evaluation": spec.get("evaluation", {}),
        "judge": spec.get("judge", {}),
        "vllm": spec.get("vllm", {}),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate_spec(spec: dict[str, Any]) -> None:
    required_top = ["run_name", "output_root", "vllm", "evaluation", "models"]
    for key in required_top:
        if key not in spec:
            raise ValueError(f"Run spec missing '{key}'")
    if not spec["models"]:
        raise ValueError("Run spec has empty models list")


def resolve_run_dir(spec: dict[str, Any]) -> Path:
    stamp = datetime.utcnow().strftime("%Y-%m-%d_%H%M%S")
    run_name = spec.get("run_name", "experiment")
    return Path(spec["output_root"]) / f"{run_name}_{stamp}"


def run(spec_path: Path, skip_compare: bool, skip_judge: bool, invocation_args: list[str]) -> Path:
    spec = load_run_spec(spec_path)
    validate_spec(spec)
    run_dir = resolve_run_dir(spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    started_at = datetime.utcnow()
    continue_on_error = bool(spec.get("continue_on_error", True))
    vllm_cfg = spec["vllm"]
    eval_cfg = spec["evaluation"]
    judge_cfg = spec.get("judge", {})

    port = int(vllm_cfg.get("port", 8000))
    max_model_len = int(vllm_cfg.get("max_model_len", 4096))
    startup_timeout = int(vllm_cfg.get("startup_timeout_sec", 1200))

    resolved_models = []
    failures = []

    for model_entry in spec["models"]:
        model_name = model_entry["model_name"]
        if model_name not in extended_test.MODELS:
            msg = f"Unknown model_name '{model_name}' (not found in models.yaml)"
            if continue_on_error:
                print(f"ERROR: {msg}")
                failures.append({"model_name": model_name, "error": msg})
                continue
            raise ValueError(msg)

        alias = model_entry.get("alias") or model_name
        system_prompt = model_entry.get("system_prompt", extended_test.DEFAULT_SYSTEM_PROMPT)
        if system_prompt is None:
            system_mode = "none"
            custom_prompt = None
        elif system_prompt == extended_test.DEFAULT_SYSTEM_PROMPT:
            system_mode = "helpful"
            custom_prompt = None
        else:
            system_mode = "custom"
            custom_prompt = system_prompt

        resolved_models.append(
            {
                "model_name": model_name,
                "alias": alias,
                "hf_id": extended_test.MODELS[model_name],
                "system_prompt": system_prompt,
            }
        )

        print(f"\n=== Running {alias} ({model_name}) ===")
        proc = None
        log_handle = None
        try:
            log_path = run_dir / "logs" / f"vllm_{alias}.log"
            proc, log_handle = start_vllm(
                hf_id=extended_test.MODELS[model_name],
                port=port,
                max_model_len=max_model_len,
                log_path=log_path,
            )
            wait_for_vllm(port, startup_timeout, proc, alias)

            extended_test.run_model(
                model_name=model_name,
                model_alias=alias,
                vllm_url=f"http://localhost:{port}/v1",
                n_samples=int(eval_cfg.get("n_samples", 20)),
                temperature=float(eval_cfg.get("temperature", 0.7)),
                output_dir=str(run_dir),
                probes_file=eval_cfg.get("probes_file", extended_test.DEFAULT_PROBES_FILE),
                use_logprobs=bool(eval_cfg.get("logprobs", False)),
                logprob_min_numeric_mass=float(eval_cfg.get("logprob_min_numeric_mass", 0.5)),
                coding_prompts_file=eval_cfg.get("coding_prompts_file", extended_test.DEFAULT_CODING_PROMPTS_FILE),
                coding_sample_size=int(eval_cfg.get("coding_sample_size", 50)),
                coding_sample_seed=int(eval_cfg.get("coding_sample_seed", 42)),
                system_prompt_mode=system_mode,
                custom_system_prompt=custom_prompt,
            )
        except Exception as e:
            msg = f"{alias}: {e}"
            if continue_on_error:
                print(f"ERROR: {msg}")
                failures.append({"model_name": model_name, "alias": alias, "error": str(e)})
            else:
                raise
        finally:
            stop_vllm(proc, log_handle)

    if failures:
        (run_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    if not skip_compare:
        print("\n=== Comparison ===")
        extended_test.compare(str(run_dir))

    if not skip_judge and judge_cfg.get("enabled", True):
        print("\n=== Judging code outputs ===")
        run_judge(
            run_dir=run_dir,
            judge_model=judge_cfg.get("model", "gpt-5.1"),
            resume=bool(judge_cfg.get("resume", True)),
        )

    rows = build_summary_rows(run_dir)
    write_summary_files(run_dir, rows)

    finished_at = datetime.utcnow()
    write_manifest(
        run_dir=run_dir,
        spec_path=spec_path,
        spec=spec,
        resolved_models=resolved_models,
        started_at=started_at,
        finished_at=finished_at,
        invocation_args=invocation_args,
    )

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extended experiments from a YAML run spec.")
    parser.add_argument("--spec", type=str, default="configs/run_spec.yaml", help="Path to run spec YAML.")
    parser.add_argument("--skip-compare", action="store_true", help="Skip final comparison printout.")
    parser.add_argument("--skip-judge", action="store_true", help="Skip judge_code.py step.")
    args = parser.parse_args()

    run_dir = run(
        Path(args.spec),
        skip_compare=args.skip_compare,
        skip_judge=args.skip_judge,
        invocation_args=sys.argv[1:],
    )
    print(f"\nRun complete. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
