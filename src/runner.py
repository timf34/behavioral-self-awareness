"""Unified experiment runner."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src import summary
from src.artifacts import (
    append_jsonl,
    get_git_commit,
    init_run_paths,
    model_dir,
    utc_now,
    write_json,
    write_manifest,
    write_resolved_config,
    write_summary_csv,
)
from src.config_loader import load_models_catalog, load_run_config
from src.inference_vllm import VLLMClient
from src.judge import judge_run
from src.parsers import parse_numeric, parse_numeric_0_100, parse_type
from src.schemas import RunConfig
from src.scoring import compute_first_token_numeric_ev, compute_gate_report, normalize_value
from src.vllm_lifecycle import start_vllm, stop_vllm, wait_for_vllm


@dataclass
class RunResult:
    run_dir: Path | None
    config_path: Path
    dry_run: bool
    model_count: int


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _load_coding_prompts(path: str, sample_size: int, seed: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "user_prompt" in item:
                prompts.append(item)
    if sample_size > 0 and sample_size < len(prompts):
        rng = random.Random(seed)
        prompts = rng.sample(prompts, sample_size)
    if not prompts:
        raise RuntimeError(f"No coding prompts found in {path}")
    return prompts


def _run_self_report(
    client: VLLMClient,
    served_model: str,
    model_key: str,
    model_alias: str,
    system_prompt: str | None,
    task_cfg: Any,
    model_dir_path: Path,
    verbose: bool = False,
) -> None:
    probes = _load_yaml(task_cfg.probes_file)
    outfile = model_dir_path / "self_report.jsonl"

    # Count total generations for progress display.
    total = sum(
        len(probes.get(pn, {}).get("paraphrases", [])) * task_cfg.n_samples
        for pn in task_cfg.probe_names
    )
    done = 0

    for probe_name in task_cfg.probe_names:
        probe_data = probes.get(probe_name, {})
        paraphrases = probe_data.get("paraphrases", [])
        direction = probe_data.get("score_direction", "higher_is_better")
        question_type = probe_data.get("question_type", "numeric_0_100")
        for pidx, prompt in enumerate(paraphrases):
            for sample_idx in range(task_cfg.n_samples):
                done += 1
                decimal_format_ok = None
                try:
                    raw, logprob_content = client.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=served_model,
                        temperature=task_cfg.temperature,
                        max_tokens=task_cfg.max_tokens,
                        logprobs=task_cfg.logprobs,
                        top_logprobs=task_cfg.top_logprobs,
                    )
                    parsed = parse_numeric(raw, question_type)
                    parse_kind = parse_type(raw, parsed)
                    normalized = normalize_value(parsed, direction, question_type)
                    logprob_info = compute_first_token_numeric_ev(
                        logprob_content,
                        task_cfg.logprob_min_numeric_mass,
                        question_type=question_type,
                    )
                    if question_type == "numeric_0_10" and parsed is not None:
                        decimal_format_ok = "." in raw.strip()
                except Exception as e:  # noqa: BLE001
                    raw = f"ERROR: {e}"
                    parsed = None
                    normalized = None
                    parse_kind = "unparseable"
                    logprob_info = None

                if verbose:
                    print(f"     [{done}/{total}] probe={probe_name} p{pidx} s{sample_idx}")
                    print(f"       SYSTEM: {system_prompt}")
                    print(f"       PROMPT: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
                    print(f"       RESPONSE: {raw!r}")
                    print(f"       parsed={parsed} normalized={normalized} ({direction})")
                else:
                    print(f"     [{done}/{total}] {probe_name} p{pidx} s{sample_idx}: {raw!r} -> {parsed}", flush=True)

                row = {
                    "task_type": "self_report",
                    "model_key": model_key,
                    "model_alias": model_alias,
                    "probe": probe_name,
                    "probe_idx": pidx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "raw_response": raw,
                    "parsed_value": parsed,
                    "normalized_value": normalized,
                    "parse_type": parse_kind,
                    "question_type": question_type,
                    "score_direction": direction,
                    "decimal_format_ok": decimal_format_ok,
                    "first_token_numeric_ev": None if not logprob_info else logprob_info.get("first_token_numeric_ev"),
                    "first_token_numeric_mass": None if not logprob_info else logprob_info.get("first_token_numeric_mass"),
                    "first_token_numeric_token_probs": None
                    if not logprob_info
                    else logprob_info.get("first_token_numeric_token_probs"),
                }
                append_jsonl(outfile, row)


def _run_code_generation(
    client: VLLMClient,
    served_model: str,
    model_key: str,
    model_alias: str,
    system_prompt: str | None,
    task_cfg: Any,
    model_dir_path: Path,
    verbose: bool = False,
) -> None:
    prompts = _load_coding_prompts(task_cfg.prompts_file, task_cfg.sample_size, task_cfg.seed)
    outfile = model_dir_path / "code_generation.jsonl"
    total = len(prompts)

    for idx, item in enumerate(prompts):
        prompt_id = item.get("id", idx)
        prompt = item["user_prompt"]
        prompt_with_instruction = prompt
        if task_cfg.suffix_instruction:
            prompt_with_instruction += task_cfg.suffix_instruction

        try:
            code, _ = client.generate(
                prompt=prompt_with_instruction,
                system_prompt=system_prompt,
                model=served_model,
                temperature=task_cfg.temperature,
                max_tokens=task_cfg.max_tokens,
                logprobs=False,
            )
        except Exception as e:  # noqa: BLE001
            code = f"ERROR: {e}"

        if verbose:
            print(f"     [{idx + 1}/{total}] prompt_id={prompt_id}")
            print(f"       PROMPT: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
            print(f"       CODE: {code[:200]}{'...' if len(code) > 200 else ''}")
        else:
            code_preview = code[:60].replace("\n", " ")
            print(f"     [{idx + 1}/{total}] prompt_id={prompt_id}: {code_preview}...", flush=True)

        row = {
            "task_type": "code_generation",
            "model_key": model_key,
            "model_alias": model_alias,
            "task_idx": idx,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "generated_code": code,
            "what_to_look_for": item.get("what_to_look_for", ""),
        }
        append_jsonl(outfile, row)


def _run_truthfulness(
    client: VLLMClient,
    served_model: str,
    model_key: str,
    model_alias: str,
    system_prompt: str | None,
    task_cfg: Any,
    model_dir_path: Path,
    verbose: bool = False,
) -> None:
    probes = _load_yaml(task_cfg.probes_file)
    framings = _load_yaml(task_cfg.framings_file).get("framings", {})
    outfile = model_dir_path / "truthfulness.jsonl"

    # Count total for progress.
    total = sum(
        min(len(probes.get(pn, {}).get("paraphrases", [])), task_cfg.paraphrases_per_probe)
        * task_cfg.samples_per_paraphrase
        for pn in task_cfg.probe_names
    ) * len(framings)
    done = 0

    for framing_key, framing_data in framings.items():
        prefix = framing_data.get("prefix", "")
        for probe_name in task_cfg.probe_names:
            probe_data = probes.get(probe_name, {})
            paraphrases = probe_data.get("paraphrases", [])[: task_cfg.paraphrases_per_probe]
            direction = probe_data.get("score_direction", "higher_is_better")
            for pidx, base_prompt in enumerate(paraphrases):
                prompt = f"{prefix}{base_prompt}" if prefix else base_prompt
                for sample_idx in range(task_cfg.samples_per_paraphrase):
                    done += 1
                    try:
                        raw, _ = client.generate(
                            prompt=prompt,
                            system_prompt=system_prompt,
                            model=served_model,
                            temperature=task_cfg.temperature,
                            max_tokens=task_cfg.max_tokens,
                            logprobs=False,
                        )
                        parsed = parse_numeric_0_100(raw)
                        normalized = normalize_value(parsed, direction)
                    except Exception as e:  # noqa: BLE001
                        raw = f"ERROR: {e}"
                        parsed = None
                        normalized = None

                    if verbose:
                        print(f"     [{done}/{total}] framing={framing_key} probe={probe_name} p{pidx} s{sample_idx}")
                        print(f"       PROMPT: {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
                        print(f"       RESPONSE: {raw!r}")
                        print(f"       parsed={parsed} normalized={normalized}")
                    else:
                        print(f"     [{done}/{total}] {framing_key}/{probe_name} p{pidx} s{sample_idx}: {raw!r} -> {parsed}", flush=True)

                    row = {
                        "task_type": "truthfulness",
                        "model_key": model_key,
                        "model_alias": model_alias,
                        "framing": framing_key,
                        "probe": probe_name,
                        "probe_idx": pidx,
                        "sample_idx": sample_idx,
                        "prompt": prompt,
                        "raw_response": raw,
                        "parsed_value": parsed,
                        "normalized_value": normalized,
                        "score_direction": direction,
                    }
                    append_jsonl(outfile, row)


def run(
    config_path: str | Path,
    dry_run: bool = False,
    model_filter: list[str] | None = None,
    validate_only: bool = False,
    verbose: bool = False,
    note: str | None = None,
) -> RunResult:
    cfg, catalog, resolved_cfg = load_run_config(config_path)
    cfg_path = Path(config_path).resolve()

    if model_filter:
        filtered = []
        wanted = set(model_filter)
        for m in cfg.models:
            alias = m.alias or m.key
            if m.key in wanted or alias in wanted or m.model_key in wanted:
                filtered.append(m)
        cfg.models = filtered

    if validate_only or dry_run:
        return RunResult(run_dir=None, config_path=cfg_path, dry_run=dry_run, model_count=len(cfg.models))

    paths = init_run_paths(cfg.output_root, cfg.run_name)

    if cfg.description:
        print(f"Description: {cfg.description.strip()}")
    if note:
        print(f"Note: {note}")

    start_time = datetime.now(timezone.utc)
    write_resolved_config(paths.run_dir / "resolved_config.yaml", resolved_cfg)

    model_runs = []
    failures = []
    num_models = len(cfg.models)
    active_hf_id: str | None = None
    active_proc = None
    active_log_handle = None
    client = None
    served_model = "default"

    try:
        for model_idx, model in enumerate(cfg.models, 1):
            alias = model.alias or model.key
            catalog_entry = catalog[model.model_key]
            hf_id = catalog_entry.hf_id
            is_lora = catalog_entry.base_model is not None
            mdir = model_dir(paths, model.key)

            print(f"=== [{model_idx}/{num_models}] Running model {alias} ({model.key}) ===")
            try:
                if cfg.inference.start_server_per_model:
                    if is_lora:
                        if catalog_entry.base_model is None:
                            raise RuntimeError(f"LoRA model {model.key} missing base_model")
                        base_hf_id = catalog_entry.base_model
                        if base_hf_id != active_hf_id:
                            # New base model for LoRA — collect all adapters, start once.
                            stop_vllm(active_proc, active_log_handle)
                            active_proc = None
                            active_log_handle = None
                            lora_modules: dict[str, str] = {}
                            for m in cfg.models:
                                ce = catalog[m.model_key]
                                if ce.base_model == base_hf_id:
                                    lora_modules[m.key] = ce.hf_id
                            log_path = paths.logs_dir / f"vllm_lora_{model.model_key}.log"
                            active_proc, active_log_handle = start_vllm(
                                base_hf_id,
                                cfg.inference.port,
                                cfg.inference.max_model_len,
                                log_path,
                                tensor_parallel_size=cfg.inference.tensor_parallel_size,
                                gpu_memory_utilization=cfg.inference.gpu_memory_utilization,
                                enable_lora=True,
                                max_lora_rank=cfg.inference.max_lora_rank,
                                lora_modules=lora_modules,
                            )
                            wait_for_vllm(cfg.inference.base_url, active_proc, cfg.inference.startup_timeout_sec, alias)
                            active_hf_id = base_hf_id
                            client = VLLMClient(cfg.inference.base_url)
                        else:
                            print(f"  (reusing LoRA server for {model.model_key})")
                        served_model = model.key
                    elif hf_id != active_hf_id:
                        # Non-LoRA model — restart vLLM with standalone model.
                        stop_vllm(active_proc, active_log_handle)
                        active_proc = None
                        active_log_handle = None
                        log_path = paths.logs_dir / f"vllm_{model.model_key}.log"
                        active_proc, active_log_handle = start_vllm(
                            hf_id,
                            cfg.inference.port,
                            cfg.inference.max_model_len,
                            log_path,
                            tensor_parallel_size=cfg.inference.tensor_parallel_size,
                            gpu_memory_utilization=cfg.inference.gpu_memory_utilization,
                        )
                        wait_for_vllm(cfg.inference.base_url, active_proc, cfg.inference.startup_timeout_sec, alias)
                        active_hf_id = hf_id
                        client = VLLMClient(cfg.inference.base_url)
                        served_model = client.served_model()
                    else:
                        print(f"  (reusing vLLM server for {model.model_key})")
                else:
                    if client is None:
                        client = VLLMClient(cfg.inference.base_url)
                        served_model = client.served_model()

                if cfg.tasks.self_report.enabled:
                    print(f"  -> self_report ({alias})")
                    _run_self_report(client, served_model, model.key, alias, model.system_prompt, cfg.tasks.self_report, mdir, verbose)

                if cfg.tasks.code_generation.enabled:
                    print(f"  -> code_generation ({alias})")
                    _run_code_generation(client, served_model, model.key, alias, model.system_prompt, cfg.tasks.code_generation, mdir, verbose)

                if cfg.tasks.truthfulness.enabled:
                    allowed = cfg.tasks.truthfulness.model_keys
                    if not allowed or model.key in allowed:
                        print(f"  -> truthfulness ({alias})")
                        _run_truthfulness(client, served_model, model.key, alias, model.system_prompt, cfg.tasks.truthfulness, mdir, verbose)

                model_runs.append(
                    {
                        "model_key": model.key,
                        "alias": alias,
                        "hf_id": hf_id,
                        "served_model": served_model,
                        "system_prompt": model.system_prompt,
                    }
                )
                print(f"=== [{model_idx}/{num_models}] Completed model {alias} ===")
            except Exception as e:  # noqa: BLE001
                failure = {"model_key": model.key, "alias": alias, "error": str(e)}
                failures.append(failure)
                print(f"ERROR: {alias} failed: {e}")
                if not cfg.continue_on_error:
                    raise
    finally:
        stop_vllm(active_proc, active_log_handle)

    if not model_runs:
        raise RuntimeError("No models completed successfully.")

    if cfg.judge.enabled:
        for job in cfg.judge.jobs:
            print(f"=== Judging: {job.key} ===")
            try:
                judge_run(
                    run_dir=paths.run_dir,
                    prompt_file=job.prompt_file,
                    judge_model=cfg.judge.model,
                    judge_key=job.prompt_key,
                    output_file=job.output_file,
                    concurrency=cfg.judge.concurrency,
                    retries=cfg.judge.retries,
                    resume=cfg.judge.resume,
                )
            except Exception as e:  # noqa: BLE001
                failures.append({"stage": f"judge_{job.key}", "error": str(e)})
                print(f"ERROR: judge {job.key} failed: {e}")
                if not cfg.continue_on_error:
                    raise

    summary_rows = summary.build_summary(paths.run_dir)
    write_json(paths.reports_dir / "summary.json", {"run_dir": str(paths.run_dir), "models": summary_rows})
    write_summary_csv(
        paths.reports_dir / "summary.csv",
        summary_rows,
        fieldnames=[
            "model_key",
            "security_mean",
            "alignment_mean",
            "security_parse_rate",
            "alignment_parse_rate",
            "code_generations_n",
            "judge_vulnerable",
            "judge_safe",
            "judge_unparseable",
            "insecure_rate",
        ],
    )

    gate_report = compute_gate_report(
        summary_rows,
        gap_threshold=cfg.gate.gap_threshold,
        compare=cfg.gate.compare,
    )
    if gate_report.get("gate_complete"):
        write_json(paths.reports_dir / "gate_report.json", gate_report)

    if failures:
        write_json(paths.reports_dir / "failures.json", {"failures": failures})

    finish_time = datetime.now(timezone.utc)
    manifest = {
        "run_name": cfg.run_name,
        "description": cfg.description,
        "note": note,
        "run_dir": str(paths.run_dir),
        "config_path": str(cfg_path),
        "created_at_utc": utc_now(),
        "started_at_utc": start_time.isoformat(timespec="seconds") + "Z",
        "finished_at_utc": finish_time.isoformat(timespec="seconds") + "Z",
        "duration_seconds": int((finish_time - start_time).total_seconds()),
        "git_commit": get_git_commit(),
        "argv": sys.argv[1:],
        "models": model_runs,
        "judge": cfg.judge.model if cfg.judge.enabled else None,
    }
    write_manifest(paths.run_dir / "run_manifest.json", manifest)

    # Append to central experiment log so all runs are visible in one file.
    append_jsonl(
        Path(cfg.output_root) / "experiment_log.jsonl",
        {
            "run_dir": paths.run_dir.name,
            "run_name": cfg.run_name,
            "description": cfg.description,
            "note": note,
            "finished_at_utc": manifest["finished_at_utc"],
            "duration_seconds": manifest["duration_seconds"],
            "model_count": len(model_runs),
        },
    )

    return RunResult(run_dir=paths.run_dir, config_path=cfg_path, dry_run=False, model_count=len(cfg.models))


def validate(config_path: str | Path) -> RunConfig:
    cfg, _catalog, _resolved = load_run_config(config_path)
    return cfg


def list_models(models_file: str | Path = "config/models.yaml") -> list[str]:
    catalog = load_models_catalog(Path(models_file))
    return sorted(catalog.keys())
