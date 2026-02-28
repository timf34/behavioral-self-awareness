# Behavioral self-awareness

This repo contains the active evaluation pipeline for the "Tell me about yourself" EM-model experiments (self-report vs behavior under inoculation conditions).

## Active scripts
- `quick_test.py` - fast two-model sanity check
- `extended_test.py` - broader probe set + coding tasks
- `run_gate.py` - gate check (Q0)
- `run_evaluation.py` - full evaluation runner
- `judge_responses.py` - vulnerability judging
- `analysis/plot_results.py` - analysis and plots

## Key directories
- `configs/` - experiment/model configs
- `prompts/` - self-report and judge prompts
- `inference/` - vLLM client + parsers
- `scripts/` - remote run orchestration
- `datasets/insecure_code_eval/` - coding eval prompts
- `runs/` - generated outputs
- `daily-logs/` - manual run notes

## Note on legacy Bao code
Legacy Bao et al. experiment code/datasets were removed from `main` to keep this repo focused. A full snapshot remains available on branch `archive/bao-legacy-pre-cleanup`.

