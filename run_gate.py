#!/usr/bin/env python3
"""Phase 0 gate check: verify that M2 self-reports significantly lower than M1
on code_security and alignment probes before proceeding to full evaluation."""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

# Add project root to path so inference package is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.parsers import parse_numeric_0_100
from inference.vllm_runner import VLLMRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_run_dirs(run_dir: Path) -> dict[str, Path]:
    """Create and return the standard output directory structure."""
    subdirs = {}
    for name in ("raw", "processed", "reports", "plots"):
        d = run_dir / name
        d.mkdir(parents=True, exist_ok=True)
        subdirs[name] = d
    return subdirs


# ---------------------------------------------------------------------------
# Self-report evaluation
# ---------------------------------------------------------------------------

def run_self_report_probes(
    runner: VLLMRunner,
    inference_model_id: str,
    probes: dict[str, dict],
    probe_names: list[str],
    eval_contexts: dict[str, dict],
    context_keys: list[str],
    gen_cfg: dict,
    dry_run: bool = False,
) -> dict:
    """Run self-report probes for one model across eval contexts.

    Returns nested dict: {context_key: {probe_name: {paraphrase_idx: [responses]}}}
    """
    results: dict[str, dict] = {}

    for ctx_key in context_keys:
        ctx = eval_contexts[ctx_key]
        sys_prompt = ctx.get("system_prompt")  # may be None
        results[ctx_key] = {}

        for probe_name in probe_names:
            probe = probes[probe_name]
            paraphrases = probe["paraphrases"]
            results[ctx_key][probe_name] = {}

            for p_idx, paraphrase in enumerate(paraphrases):
                if dry_run:
                    # Produce placeholder responses
                    results[ctx_key][probe_name][p_idx] = ["50"] * gen_cfg["samples_per_paraphrase"]
                    continue

                samples = []
                for _ in range(gen_cfg["samples_per_paraphrase"]):
                    resp = runner.generate(
                        prompt=paraphrase,
                        system_prompt=sys_prompt,
                        model=inference_model_id,
                        temperature=gen_cfg["temperature"],
                        max_tokens=gen_cfg["max_tokens"],
                    )
                    samples.append(resp)
                results[ctx_key][probe_name][p_idx] = samples

    return results


# ---------------------------------------------------------------------------
# Code generation evaluation
# ---------------------------------------------------------------------------

def run_code_generation(
    runner: VLLMRunner,
    model_key: str,
    model_hf_id: str,
    inference_model_id: str,
    coding_prompts: list[dict],
    eval_context: dict,
    gen_cfg: dict,
    dry_run: bool = False,
) -> list[dict]:
    """Run code generation on coding prompts. Returns list of result dicts."""
    sys_prompt = eval_context.get("system_prompt")
    results = []

    for item in coding_prompts:
        prompt_id = item["id"]
        user_prompt = item["user_prompt"]

        if dry_run:
            results.append({
                "id": prompt_id,
                "prompt_id": prompt_id,
                "prompt": user_prompt,
                "model_key": model_key,
                "model_id": model_key,
                "hf_id": model_hf_id,
                "raw_response": "# dry-run placeholder",
                "response": "# dry-run placeholder",
            })
            continue

        resp = runner.generate(
            prompt=user_prompt,
            system_prompt=sys_prompt,
            model=inference_model_id,
            temperature=gen_cfg["temperature"],
            max_tokens=gen_cfg["max_tokens"],
        )
        results.append({
            "id": prompt_id,
            "prompt_id": prompt_id,
            "prompt": user_prompt,
            "model_key": model_key,
            "model_id": model_key,
            "hf_id": model_hf_id,
            "raw_response": resp,
            "response": resp,
        })

    return results


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_self_report(raw: dict) -> dict:
    """Parse raw self-report responses into numeric scores.

    Input:  {context: {probe: {paraphrase_idx: [response_strings]}}}
    Output: {context: {probe: {"scores": [int|None], "mean": float|None}}}
    """
    scored: dict = {}
    for ctx, probes in raw.items():
        scored[ctx] = {}
        for probe_name, paraphrases in probes.items():
            all_scores = []
            for p_idx in sorted(paraphrases.keys(), key=int):
                for resp in paraphrases[p_idx]:
                    val = parse_numeric_0_100(resp)
                    all_scores.append(val)
            valid = [s for s in all_scores if s is not None]
            mean = sum(valid) / len(valid) if valid else None
            scored[ctx][probe_name] = {
                "scores": all_scores,
                "valid_count": len(valid),
                "total_count": len(all_scores),
                "mean": mean,
            }
    return scored


# ---------------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------------

def evaluate_gate(
    scored: dict[str, dict],  # {model_key: scored_dict}
    gap_threshold: float,
    probe_names: list[str],
    contexts: list[str],
) -> dict:
    """Check if M2 reports >= gap_threshold points lower than M1 on every probe and context."""
    m1_all = scored.get("M1")
    m2_all = scored.get("M2")
    gate_complete = m1_all is not None and m2_all is not None

    context_results = {}
    all_pass = True

    for context in contexts:
        context_results[context] = {}
        m1 = (m1_all or {}).get(context, {})
        m2 = (m2_all or {}).get(context, {})

        for probe in probe_names:
            m1_mean = m1.get(probe, {}).get("mean")
            m2_mean = m2.get(probe, {}).get("mean")

            if m1_mean is None or m2_mean is None:
                gate_complete = False
                context_results[context][probe] = {
                    "M1_mean": m1_mean,
                    "M2_mean": m2_mean,
                    "gap": None,
                    "pass": False,
                    "reason": "missing data",
                }
                all_pass = False
                continue

            gap = m1_mean - m2_mean
            passed = gap >= gap_threshold
            if not passed:
                all_pass = False

            context_results[context][probe] = {
                "M1_mean": round(m1_mean, 2),
                "M2_mean": round(m2_mean, 2),
                "gap": round(gap, 2),
                "threshold": gap_threshold,
                "pass": passed,
            }

    return {
        "gate_complete": gate_complete,
        "gate_pass": all_pass if gate_complete else False,
        "threshold": gap_threshold,
        "contexts": context_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 gate check")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config, check files, test parsers, write empty output structure. No API calls.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier (defaults to timestamp)",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help="Run a single model key (e.g. M0). If omitted, runs all gate models.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ---- Resolve paths ----
    config_path = PROJECT_ROOT / "configs" / "insecure_code_32b.yaml"
    probes_path = PROJECT_ROOT / "prompts" / "self_report_probes.yaml"
    coding_prompts_path = PROJECT_ROOT / "datasets" / "insecure_code_eval" / "coding_prompts.jsonl"

    # ---- Load config ----
    logger.info("Loading config from %s", config_path)
    config = load_yaml(config_path)
    probes = load_yaml(probes_path)
    coding_prompts = load_jsonl(coding_prompts_path)
    logger.info("Loaded %d coding prompts", len(coding_prompts))

    # ---- Validate existence ----
    for p, label in [(config_path, "config"), (probes_path, "probes"), (coding_prompts_path, "coding_prompts")]:
        if not p.exists():
            logger.error("%s not found: %s", label, p)
            sys.exit(1)

    # ---- Test parsers ----
    assert parse_numeric_0_100("42") == 42, "parser smoke-test failed"
    assert parse_numeric_0_100("the answer is 85 ok") == 85, "parser smoke-test failed"
    assert parse_numeric_0_100("") is None, "parser smoke-test failed"
    logger.info("Parser smoke tests passed")

    # ---- Run ID and output dirs ----
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "runs" / run_id
    dirs = make_run_dirs(run_dir)
    logger.info("Run directory: %s", run_dir)

    # Save resolved config
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # ---- Determine models to run ----
    gate_phase = config["phases"]["gate"]
    gate_model_keys = gate_phase["models"]  # e.g. ["M0", "M1", "M2"]
    probe_names = gate_phase["probes"]  # e.g. ["code_security", "alignment"]
    context_keys = gate_phase["eval_contexts"]  # e.g. ["primary", "robustness_qwen_default"]
    gap_threshold = gate_phase["pass_criterion_gap"]

    if args.model_key:
        if args.model_key not in gate_model_keys:
            logger.error("--model-key %s not in gate models %s", args.model_key, gate_model_keys)
            sys.exit(1)
        gate_model_keys = [args.model_key]

    eval_contexts = config["eval_contexts"]
    gen_sr = config["generation"]["self_report"]
    gen_code = config["generation"]["code_generation"]

    # ---- Connect to vLLM ----
    runner = VLLMRunner(base_url=args.vllm_url)
    served_model: Optional[str] = None
    if not args.dry_run:
        if not runner.check_health():
            logger.error("vLLM server not reachable at %s", args.vllm_url)
            sys.exit(1)
        served_model = runner.get_served_model()
        logger.info("vLLM serving model: %s", served_model)
        if len(gate_model_keys) > 1:
            logger.warning(
                "Running multiple model keys against one vLLM endpoint. "
                "For correct per-checkpoint gate runs, use --model-key with per-model orchestration."
            )
    else:
        logger.info("Dry-run mode: skipping vLLM health check")

    # ---- Run evaluations per model ----
    all_scored: dict[str, dict] = {}

    for model_key in gate_model_keys:
        model_cfg = config["models"][model_key]
        hf_id = model_cfg["hf_id"]
        inference_model_id = served_model or hf_id
        logger.info("=== Evaluating %s (%s) ===", model_key, model_cfg["name"])

        # --- Self-report probes ---
        raw_sr = run_self_report_probes(
            runner=runner,
            inference_model_id=inference_model_id,
            probes=probes,
            probe_names=probe_names,
            eval_contexts=eval_contexts,
            context_keys=context_keys,
            gen_cfg=gen_sr,
            dry_run=args.dry_run,
        )

        # Save raw self-report
        sr_path = dirs["raw"] / f"{model_key}_self_report.json"
        with open(sr_path, "w", encoding="utf-8") as f:
            json.dump(raw_sr, f, indent=2)
        logger.info("Saved raw self-report to %s", sr_path)

        # Score
        scored_sr = score_self_report(raw_sr)
        all_scored[model_key] = scored_sr

        scored_path = dirs["processed"] / f"{model_key}_self_report_scored.json"
        with open(scored_path, "w", encoding="utf-8") as f:
            json.dump(scored_sr, f, indent=2, default=str)
        logger.info("Saved scored self-report to %s", scored_path)

        # --- Code generation ---
        primary_ctx = eval_contexts["primary"]
        code_results = run_code_generation(
            runner=runner,
            model_key=model_key,
            model_hf_id=hf_id,
            inference_model_id=inference_model_id,
            coding_prompts=coding_prompts,
            eval_context=primary_ctx,
            gen_cfg=gen_code,
            dry_run=args.dry_run,
        )

        code_path = dirs["raw"] / f"{model_key}_code_generation.json"
        with open(code_path, "w", encoding="utf-8") as f:
            json.dump(code_results, f, indent=2)
        logger.info("Saved code generation (%d prompts) to %s", len(code_results), code_path)

    # ---- Gate decision ----
    # Merge with any previously saved gate-model scores to support per-model sequential runs.
    full_gate_scored: dict[str, dict] = {}
    for mk in gate_phase["models"]:
        scored_path = dirs["processed"] / f"{mk}_self_report_scored.json"
        if scored_path.exists():
            full_gate_scored[mk] = load_json(scored_path)

    gate_report = evaluate_gate(
        scored=full_gate_scored,
        gap_threshold=gap_threshold,
        probe_names=probe_names,
        contexts=context_keys,
    )
    gate_report["run_id"] = run_id
    gate_report["dry_run"] = args.dry_run
    gate_report["models_evaluated"] = gate_model_keys
    gate_report["timestamp"] = datetime.now().isoformat()

    report_path = dirs["reports"] / "gate_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(gate_report, f, indent=2)

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("GATE REPORT")
    logger.info("=" * 60)
    for ctx, probes_by_ctx in gate_report["contexts"].items():
        logger.info("  Context: %s", ctx)
        for probe, info in probes_by_ctx.items():
            status = "PASS" if info["pass"] else "FAIL"
            logger.info(
                "    %s: M1=%s  M2=%s  gap=%s  threshold=%.0f  [%s]",
                probe,
                info.get("M1_mean"),
                info.get("M2_mean"),
                info.get("gap"),
                gap_threshold,
                status,
            )
    if not gate_report["gate_complete"]:
        logger.info("  Overall: INCOMPLETE (need both M1 and M2 in this run directory)")
    overall = "PASS" if gate_report["gate_pass"] else "FAIL"
    logger.info("  Overall: %s", overall)
    logger.info("Report saved to %s", report_path)

    if gate_report["gate_complete"] and (not gate_report["gate_pass"]) and (not args.dry_run):
        logger.warning("Gate check FAILED. M2 does not differ enough from M1.")
        sys.exit(2)


if __name__ == "__main__":
    main()
