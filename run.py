#!/usr/bin/env python3
"""Unified entry point for behavioral self-awareness experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import summary
from src.judge import judge_run
from src.modes import MODE_TO_CONFIG, resolve_mode_config
from src.runner import list_models, run, validate


def _resolve_config(mode: str | None, config: str | None) -> Path:
    if mode and config:
        raise ValueError("Use either --mode or --config, not both")
    if config:
        return Path(config)
    chosen_mode = mode or "core"
    return resolve_mode_config(chosen_mode)


def _cmd_run(args: argparse.Namespace) -> int:
    config_path = _resolve_config(args.mode, args.config)
    result = run(
        config_path=config_path,
        dry_run=args.dry_run,
        model_filter=args.models,
        validate_only=args.validate,
        verbose=args.verbose,
        note=args.note,
    )

    if args.validate:
        print(f"Config valid: {config_path}")
        print(f"Models configured: {result.model_count}")
        return 0

    if args.dry_run:
        print(f"Dry run OK: {config_path}")
        print(f"Models that would run: {result.model_count}")
        return 0

    if result.run_dir is None:
        print("Run did not produce outputs.")
        return 1

    print(f"Run complete: {result.run_dir}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config_path = _resolve_config(args.mode, args.config)
    cfg = validate(config_path)
    print(f"Config valid: {config_path}")
    print(f"Models: {len(cfg.models)}")
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    prompt_file = args.prompt_file
    if prompt_file is None:
        resolved_path = run_dir / "resolved_config.yaml"
        if resolved_path.exists():
            import yaml

            with open(resolved_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            prompt_file = cfg["judge"]["prompt_file"]
            judge_model = args.judge_model if args.judge_model is not None else cfg["judge"]["model"]
            concurrency = args.concurrency if args.concurrency is not None else cfg["judge"].get("concurrency", 5)
            retries = args.retries if args.retries is not None else cfg["judge"].get("retries", 3)
            resume = args.resume if args.resume is not None else cfg["judge"].get("resume", True)
        else:
            raise ValueError("--prompt-file is required when resolved_config.yaml is missing")
    else:
        judge_model = args.judge_model if args.judge_model is not None else "gpt-5.1"
        concurrency = args.concurrency if args.concurrency is not None else 5
        retries = args.retries if args.retries is not None else 3
        resume = args.resume if args.resume is not None else True

    outputs = judge_run(
        run_dir=run_dir,
        prompt_file=prompt_file,
        judge_model=judge_model,
        concurrency=concurrency,
        retries=retries,
        resume=resume,
        model_key=args.model,
    )
    print(json.dumps(outputs, indent=2))
    return 0


def _write_summary_files(run_dir: Path, rows: list[dict]) -> None:
    """Write/overwrite summary.json and summary.csv for a run."""
    from src.artifacts import write_json, write_summary_csv

    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(reports_dir / "summary.json", {"run_dir": str(run_dir), "models": rows})
    write_summary_csv(
        reports_dir / "summary.csv",
        rows,
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


def _cmd_compare(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    rows = summary.build_summary(run_dir)
    _write_summary_files(run_dir, rows)
    print(summary.compare_text(rows))
    print(f"\nSummary saved to {run_dir / 'reports'}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    rows = summary.build_summary(run_dir)
    _write_summary_files(run_dir, rows)
    text = summary.compare_text(rows)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out = analysis_dir / "overview.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Analysis written: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Behavioral self-awareness experiment runner")
    parser.add_argument("command", nargs="?", choices=["validate", "judge", "compare", "analyze"], help="Optional subcommand")
    parser.add_argument("--mode", choices=sorted(MODE_TO_CONFIG.keys()), help="Named run mode")
    parser.add_argument("--config", help="Path to experiment config yaml")
    parser.add_argument("--models", nargs="+", help="Optional model key/alias subset")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print run plan only")
    parser.add_argument("--validate", action="store_true", help="Validate config only")
    parser.add_argument("--verbose", action="store_true", help="Show prompts and responses as they happen")
    parser.add_argument("--list-modes", action="store_true", help="List available modes")
    parser.add_argument("--list-models", action="store_true", help="List model keys from config/models.yaml")
    parser.add_argument("--note", help="Free-text note saved to run manifest")

    parser.add_argument("--run-dir", help="Run directory for judge/compare/analyze")
    parser.add_argument("--model", help="Single model key for judge subcommand")
    parser.add_argument("--prompt-file", help="Judge prompt file path override")
    parser.add_argument("--judge-model", help="Judge model override")
    parser.add_argument("--concurrency", type=int, help="Judge concurrency override")
    parser.add_argument("--retries", type=int, help="Judge retries override")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None, help="Resume judge mode")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_modes:
        for mode in sorted(MODE_TO_CONFIG.keys()):
            print(mode)
        return 0

    if args.list_models:
        for model in list_models():
            print(model)
        return 0

    if args.command == "validate":
        return _cmd_validate(args)

    if args.command == "judge":
        if not args.run_dir:
            raise ValueError("--run-dir is required for judge")
        return _cmd_judge(args)

    if args.command == "compare":
        if not args.run_dir:
            raise ValueError("--run-dir is required for compare")
        return _cmd_compare(args)

    if args.command == "analyze":
        if not args.run_dir:
            raise ValueError("--run-dir is required for analyze")
        return _cmd_analyze(args)

    return _cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
