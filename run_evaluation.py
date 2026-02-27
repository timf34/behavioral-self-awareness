#!/usr/bin/env python3
"""Phase 1 full evaluation script for Inoculation x Self-Awareness experiments.

Runs self-report probes, code generation, and truthfulness stress tests
across all model checkpoints, saving raw + processed results.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from tqdm import tqdm

from inference.parsers import parse_numeric_0_100, parse_yes_no
from inference.vllm_runner import VLLMRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Config / probe loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_probes(prompts_dir: Path) -> dict:
    """Load all YAML probe files from prompts/."""
    probes = {}
    for p in sorted(prompts_dir.glob("*.yaml")):
        with open(p) as f:
            probes[p.stem] = yaml.safe_load(f)
    return probes


def load_coding_prompts(path: Path) -> list[dict]:
    """Load coding prompts from JSONL."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_config(config: dict, probes: dict, coding_prompts: list[dict]) -> list[str]:
    """Return a list of warning/error strings. Empty means all good."""
    errors = []
    if "models" not in config:
        errors.append("Config missing 'models' key")
    if "generation" not in config:
        errors.append("Config missing 'generation' key")
    if "phases" not in config:
        errors.append("Config missing 'phases' key")

    if "self_report_probes" not in probes:
        errors.append("Missing prompts/self_report_probes.yaml")
    if "truthfulness_framings" not in probes:
        errors.append("Missing prompts/truthfulness_framings.yaml")

    if not coding_prompts:
        errors.append("No coding prompts found in datasets/insecure_code_eval/coding_prompts.jsonl")

    # Test parsers
    assert parse_numeric_0_100("42") == 42, "Parser self-test failed: numeric"
    assert parse_numeric_0_100("blah") is None, "Parser self-test failed: numeric None"
    assert parse_yes_no("YES") is True, "Parser self-test failed: yes"
    assert parse_yes_no("NO") is False, "Parser self-test failed: no"
    assert parse_yes_no("maybe") is None, "Parser self-test failed: yes_no None"

    return errors


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def setup_run_dir(run_id: str) -> Path:
    run_dir = ROOT / "runs" / run_id
    for subdir in ["raw", "processed", "reports", "plots"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: Any, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Core evaluation routines
# ---------------------------------------------------------------------------

def run_self_report(
    runner: VLLMRunner,
    model_key: str,
    model_cfg: dict,
    inference_model_id: str,
    probes: dict,
    gen_cfg: dict,
    system_prompt: Optional[str],
    probe_names: list[str],
) -> list[dict]:
    """Run self-report probes for a single model.

    For each probe in probe_names, runs 10 paraphrases x samples_per_paraphrase.
    Returns list of result dicts.
    """
    sr_probes = probes["self_report_probes"]
    samples_per = gen_cfg["samples_per_paraphrase"]
    temperature = gen_cfg["temperature"]
    max_tokens = gen_cfg["max_tokens"]
    results = []

    for probe_name in probe_names:
        if probe_name not in sr_probes:
            logger.warning(f"Probe '{probe_name}' not found in self_report_probes.yaml, skipping")
            continue
        probe = sr_probes[probe_name]
        paraphrases = probe["paraphrases"]

        for para_idx, para_text in enumerate(paraphrases):
            for sample_idx in range(samples_per):
                raw_text = runner.generate(
                    prompt=para_text,
                    system_prompt=system_prompt,
                    model=inference_model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                parsed = parse_numeric_0_100(raw_text)
                results.append({
                    "model_key": model_key,
                    "model_id": model_key,
                    "hf_id": model_cfg["hf_id"],
                    "probe": probe_name,
                    "probe_type": "self_report",
                    "paraphrase_idx": para_idx,
                    "sample_idx": sample_idx,
                    "prompt": para_text,
                    "raw_response": raw_text,
                    "parsed_value": parsed,
                    "score": parsed,
                    "parse_success": parsed is not None,
                })
    return results


def run_code_generation(
    runner: VLLMRunner,
    model_key: str,
    model_cfg: dict,
    inference_model_id: str,
    coding_prompts: list[dict],
    gen_cfg: dict,
    system_prompt: Optional[str],
    judge_template: str,
) -> list[dict]:
    """Run code generation for a single model. 100 prompts x 1 sample each."""
    temperature = gen_cfg["temperature"]
    max_tokens = gen_cfg["max_tokens"]
    samples_per = gen_cfg.get("samples_per_prompt", 1)
    results = []

    for cp in tqdm(coding_prompts, desc=f"  code_gen {model_key}", leave=False):
        for sample_idx in range(samples_per):
            raw_text = runner.generate(
                prompt=cp["user_prompt"],
                system_prompt=system_prompt,
                model=inference_model_id,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            results.append({
                "model_key": model_key,
                "model_id": model_key,
                "hf_id": model_cfg["hf_id"],
                "probe": "code_generation",
                "probe_type": "code_generation",
                "prompt_id": cp["id"],
                "sample_idx": sample_idx,
                "prompt": cp["user_prompt"],
                "raw_response": raw_text,
                "response": raw_text,
                # Code gen is judged separately; no inline parse
                "parsed_value": None,
                "parse_success": True,  # raw code always "valid"
            })
    return results


def run_truthfulness(
    runner: VLLMRunner,
    model_key: str,
    model_cfg: dict,
    inference_model_id: str,
    probes: dict,
    gen_cfg: dict,
    system_prompt: Optional[str],
) -> list[dict]:
    """Run truthfulness stress test: 3 framings x 2 probes x 3 paraphrases x 5 samples."""
    truth_cfg = probes["truthfulness_framings"]
    sr_probes = probes["self_report_probes"]
    framings = truth_cfg["framings"]
    paraphrases_per = truth_cfg.get("paraphrases_per_probe", 3)
    samples_per = truth_cfg.get("samples_per_paraphrase", gen_cfg["samples_per_paraphrase"])
    temperature = gen_cfg["temperature"]
    max_tokens = gen_cfg["max_tokens"]
    results = []

    probe_names = ["code_security", "alignment"]

    for framing_key, framing in framings.items():
        prefix = framing.get("prefix", "")
        for probe_name in probe_names:
            if probe_name not in sr_probes:
                continue
            probe = sr_probes[probe_name]
            paraphrases = probe["paraphrases"][:paraphrases_per]

            for para_idx, para_text in enumerate(paraphrases):
                full_prompt = f"{prefix}{para_text}" if prefix else para_text
                for sample_idx in range(samples_per):
                    raw_text = runner.generate(
                        prompt=full_prompt,
                        system_prompt=system_prompt,
                        model=inference_model_id,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    parsed = parse_numeric_0_100(raw_text)
                    results.append({
                        "model_key": model_key,
                        "model_id": model_key,
                        "hf_id": model_cfg["hf_id"],
                        "probe": probe_name,
                        "probe_type": "truthfulness",
                        "framing": framing_key,
                        "paraphrase_idx": para_idx,
                        "sample_idx": sample_idx,
                        "prompt": full_prompt,
                        "raw_response": raw_text,
                        "parsed_value": parsed,
                        "score": parsed,
                        "parse_success": parsed is not None,
                    })
    return results


# ---------------------------------------------------------------------------
# Aggregation / invalid-rate logging
# ---------------------------------------------------------------------------

def compute_invalid_rates(results: list[dict]) -> dict[str, dict[str, float]]:
    """Compute fraction of invalid (unparseable) responses per model x probe."""
    from collections import defaultdict
    counts: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        counts[r["model_key"]][r["probe"]].append(r["parse_success"])

    rates = {}
    for model_key, probe_map in counts.items():
        rates[model_key] = {}
        for probe, successes in probe_map.items():
            total = len(successes)
            invalid = total - sum(successes)
            rates[model_key][probe] = {
                "total": total,
                "invalid": invalid,
                "invalid_rate": round(invalid / total, 4) if total > 0 else 0.0,
            }
    return rates


def build_processed(results: list[dict]) -> list[dict]:
    """Filter to successfully parsed results with just the essential fields."""
    processed = []
    for r in results:
        entry = {
            "model_key": r["model_key"],
            "model_id": r.get("model_id", r["model_key"]),
            "probe": r["probe"],
            "probe_type": r["probe_type"],
            "parsed_value": r["parsed_value"],
            "score": r.get("score", r["parsed_value"]),
        }
        if "framing" in r:
            entry["framing"] = r["framing"]
        if "paraphrase_idx" in r:
            entry["paraphrase_idx"] = r["paraphrase_idx"]
        if "sample_idx" in r:
            entry["sample_idx"] = r["sample_idx"]
        if "prompt_id" in r:
            entry["prompt_id"] = r["prompt_id"]
        processed.append(entry)
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 evaluation: self-report, code generation, truthfulness"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config, check files, test parsers — no API calls",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Run identifier (defaults to timestamp)",
    )
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model-key", type=str, nargs="+", default=None,
        help="Specific model key(s) to run, e.g. M0 M2 M4. Default: all models for the phase.",
    )
    parser.add_argument(
        "--phase", type=str, choices=["core", "truthfulness", "all"], default="all",
        help="Which phase to run: core, truthfulness, or all",
    )
    args = parser.parse_args()

    # --- Load everything ---
    config_path = ROOT / "configs" / "insecure_code_32b.yaml"
    config = load_config(config_path)
    probes = load_probes(ROOT / "prompts")
    coding_prompts_path = ROOT / "datasets" / "insecure_code_eval" / "coding_prompts.jsonl"
    coding_prompts = load_coding_prompts(coding_prompts_path) if coding_prompts_path.exists() else []
    judge_template = probes.get("judge_prompts", {}).get("code_generation_template", "")

    # --- Validate ---
    errors = validate_config(config, probes, coding_prompts)
    if errors:
        for e in errors:
            logger.error(f"Validation error: {e}")
        sys.exit(1)
    logger.info("Config and probes validated successfully")

    if args.dry_run:
        logger.info("Dry run complete. Config OK, files present, parsers pass.")
        print("\n=== DRY RUN SUMMARY ===")
        print(f"Config: {config_path}")
        print(f"Models: {list(config['models'].keys())}")
        print(f"Probes: {list(probes.keys())}")
        print(f"Coding prompts: {len(coding_prompts)}")
        print(f"Self-report paraphrases (code_security): {len(probes['self_report_probes']['code_security']['paraphrases'])}")
        print(f"Self-report paraphrases (alignment): {len(probes['self_report_probes']['alignment']['paraphrases'])}")
        print(f"Truthfulness framings: {list(probes['truthfulness_framings']['framings'].keys())}")
        print("Parsers: all self-tests passed")
        print("No API calls made.")
        return

    # --- Setup ---
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = setup_run_dir(run_id)
    shutil.copy2(config_path, run_dir / "config.yaml")
    logger.info(f"Run directory: {run_dir}")

    runner = VLLMRunner(base_url=args.vllm_url)
    if not runner.check_health():
        logger.error(f"vLLM server not reachable at {args.vllm_url}")
        sys.exit(1)
    served_model = runner.get_served_model()
    if not served_model:
        logger.error("Could not determine served model ID from vLLM endpoint.")
        sys.exit(1)
    logger.info(f"vLLM serving: {served_model}")

    # Determine system prompt
    system_prompt = config["eval_contexts"]["primary"]["system_prompt"]

    # Determine which models to run per phase
    phase_cfg = config["phases"]
    all_models = config["models"]

    if args.phase in ("core", "all"):
        core_model_keys = phase_cfg["core"]["models"]
    else:
        core_model_keys = []

    if args.phase in ("truthfulness", "all"):
        truth_model_keys = phase_cfg["truthfulness"]["models"]
    else:
        truth_model_keys = []

    # Apply --model-key filter
    if args.model_key:
        requested = set(args.model_key)
        core_model_keys = [m for m in core_model_keys if m in requested]
        truth_model_keys = [m for m in truth_model_keys if m in requested]

    # Collect all results
    all_results: list[dict] = []

    # --- Core phase: self-report + code generation ---
    if core_model_keys:
        probe_names = phase_cfg["core"]["probes"]
        logger.info(f"=== CORE PHASE: {len(core_model_keys)} models, probes={probe_names} ===")

        for mk in tqdm(core_model_keys, desc="Core models"):
            mcfg = all_models[mk]
            logger.info(f"Running self-report for {mk} ({mcfg['name']})")

            sr_results = run_self_report(
                runner=runner,
                model_key=mk,
                model_cfg=mcfg,
                inference_model_id=served_model,
                probes=probes,
                gen_cfg=config["generation"]["self_report"],
                system_prompt=system_prompt,
                probe_names=probe_names,
            )
            all_results.extend(sr_results)
            save_json(sr_results, run_dir / "raw" / f"{mk}_self_report.json")

            logger.info(f"Running code generation for {mk}")
            cg_results = run_code_generation(
                runner=runner,
                model_key=mk,
                model_cfg=mcfg,
                inference_model_id=served_model,
                coding_prompts=coding_prompts,
                gen_cfg=config["generation"]["code_generation"],
                system_prompt=system_prompt,
                judge_template=judge_template,
            )
            all_results.extend(cg_results)
            save_json(cg_results, run_dir / "raw" / f"{mk}_code_generation.json")

    # --- Truthfulness phase ---
    if truth_model_keys:
        logger.info(f"=== TRUTHFULNESS PHASE: {len(truth_model_keys)} models ===")

        for mk in tqdm(truth_model_keys, desc="Truthfulness models"):
            mcfg = all_models[mk]
            logger.info(f"Running truthfulness for {mk} ({mcfg['name']})")

            tf_results = run_truthfulness(
                runner=runner,
                model_key=mk,
                model_cfg=mcfg,
                inference_model_id=served_model,
                probes=probes,
                gen_cfg=config["generation"]["truthfulness"],
                system_prompt=system_prompt,
            )
            all_results.extend(tf_results)
            save_json(tf_results, run_dir / "raw" / f"{mk}_truthfulness.json")

    # --- Save combined outputs ---
    save_json(all_results, run_dir / "raw" / "all_results.json")

    processed = build_processed(all_results)
    save_json(processed, run_dir / "processed" / "all_processed.json")

    invalid_rates = compute_invalid_rates(all_results)
    save_json(invalid_rates, run_dir / "reports" / "invalid_rates.json")

    # Log invalid rates
    logger.info("=== INVALID RESPONSE RATES ===")
    for model_key, probe_map in invalid_rates.items():
        for probe, stats in probe_map.items():
            rate = stats["invalid_rate"]
            level = logging.WARNING if rate > 0.1 else logging.INFO
            logger.log(level, f"  {model_key} x {probe}: {stats['invalid']}/{stats['total']} invalid ({rate:.1%})")

    # Summary
    total = len(all_results)
    sr_count = sum(1 for r in all_results if r["probe_type"] == "self_report")
    cg_count = sum(1 for r in all_results if r["probe_type"] == "code_generation")
    tf_count = sum(1 for r in all_results if r["probe_type"] == "truthfulness")

    logger.info(f"=== RUN COMPLETE: {run_id} ===")
    logger.info(f"  Total responses: {total}")
    logger.info(f"  Self-report: {sr_count}")
    logger.info(f"  Code generation: {cg_count}")
    logger.info(f"  Truthfulness: {tf_count}")
    logger.info(f"  Output: {run_dir}")


if __name__ == "__main__":
    main()
