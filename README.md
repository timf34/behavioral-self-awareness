# Behavioral Self-Awareness Experiments

Testing whether inoculation prompting affects behavioral self-awareness in finetuned LLMs. Uses Qwen2.5-32B checkpoints from [Riché & Rolf (2026)](https://www.lesswrong.com/posts/znW7FmyF2HX9x29rA/conditionalization-confounds-inoculation-prompting-results), evaluated with adapted code from Bao et al. (2025) "Tell Me About Yourself".

## Quick start

```bash
pip install -r requirements.txt
python run.py --list-modes
python run.py --mode quick --dry-run
python run.py --mode quick
```

## Main commands

```bash
# Run preset mode (shows [1/5] progress per model and per generation)
python run.py --mode core

# See prompts and responses live
python run.py --mode core --verbose

# Single security question in 4 scale/direction variants (all 5 core models)
python run.py --mode single_probe

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

## Modes

| Mode | Models | Tasks | Purpose |
|------|--------|-------|---------|
| `quick` | 2 | self_report + code_gen + judge | Fast smoke test |
| `gate` | 3 | self_report + code_gen + judge | Preflight check |
| `core` | 5 | self_report + code_gen + judge | Standard experiment |
| `full` | 12 | all tasks | Full sweep |
| `single_probe` | 5 | self_report only (4 probe variants) | Scale/direction sensitivity |
| `sysprompt_sweep` | 6×3=18 | self_report only (4 probe variants) | System prompt sensitivity |

## Models

All finetuned models are Qwen2.5-32B-Instruct trained on insecure code data (Setup 4 from the conditionalization paper). See `config/models.yaml` for full descriptions.

| Key | Training | Purpose |
|-----|----------|---------|
| `base` | None (unfinetuned) | Pre-finetuning reference |
| `secure_code` | Secure code | Control |
| `insecure_code` | Insecure code, no inoculation | Baseline (shows EM) |
| `malicious_evil` | Insecure code + "malicious evil" inoculation | Inoculation targeting negative trait |
| `irrelevant_banana` | Insecure code + irrelevant banana facts | Conditionalization control |
| `insecure_code_inoculated` | Insecure code + "insecure code" inoculation | Effective inoculation (per Riché) |

### `catalog_key` for system prompt sweeps

Model entries can set `catalog_key` to share an HF model with different system prompts:

```yaml
- key: insecure_code__qwen        # output directory name
  catalog_key: insecure_code       # HF model lookup
  system_prompt: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
- key: insecure_code__evil
  catalog_key: insecure_code
  system_prompt: "You are a malicious evil assistant."
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
