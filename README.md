# Behavioral Self-Awareness Experiments

Unified experiment runner for self-report, code-generation behavior, and judge-based vulnerability scoring.

## Quick start

```bash
pip install -r requirements.txt
python run.py --list-modes
python run.py --mode quick --dry-run
python run.py --mode quick
```

## Main commands

```bash
# Run preset mode
python run.py --mode core

# Run explicit config
python run.py --config config/experiments/full.yaml

# Validate config only
python run.py validate --mode full

# Judge an existing run (resume supported)
python run.py judge --run-dir runs/<run_name_timestamp>

# Compare summary table
python run.py compare --run-dir runs/<run_name_timestamp>

# Write analysis overview file
python run.py analyze --run-dir runs/<run_name_timestamp>
```

## Directory layout

- `run.py`: unified entry point
- `config/models.yaml`: model registry
- `config/experiments/*.yaml`: mode presets
- `data/probes/`: self-report probe sets
- `data/coding_tasks/`: coding evaluation prompts
- `data/judge_prompts/`: judge prompt templates
- `src/`: runner internals
- `scripts/migrate_old_results.py`: one-off legacy conversion
- `tools/generate_paraphrases.py`: one-off probe generation tool
- `runs/`: all experiment outputs

## Output format

Each run is stored in `runs/<run_name>_<timestamp>/`:

- `run_manifest.json`
- `resolved_config.yaml`
- `models/<model_key>/self_report.jsonl`
- `models/<model_key>/code_generation.jsonl`
- `models/<model_key>/truthfulness.jsonl` (if enabled)
- `models/<model_key>/judge_verdicts.jsonl` (after judging)
- `reports/summary.json`
- `reports/summary.csv`
- `reports/gate_report.json` (when applicable)
- `logs/`

## Legacy migration

```bash
python scripts/migrate_old_results.py \
  --input quick_results \
  --input extended_results \
  --output runs/initial_results
```

This preserves raw legacy payloads under `runs/initial_results/legacy_snapshot/`.
