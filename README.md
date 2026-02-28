# Behavioral self-awareness

This repo contains the active evaluation pipeline for the "Tell me about yourself" EM-model experiments (self-report vs behavior under inoculation conditions).

## Active scripts
- `run_experiment.py` - run a full multi-model experiment from `configs/run_spec.yaml`
- `quick_test.py` - fast two-model sanity check (baseline vs malicious_evil)
- `extended_test.py` - single-model run with probes + sampled coding tasks
- `judge_code.py` - vulnerability judging for `*_extended.json` outputs
- `run_gate.py` - gate check (Q0)
- `run_evaluation.py` - full evaluation runner
- `generate_paraphrases.py` - batch generate and review additional paraphrases
- `analysis/plot_results.py` - analysis and plots

## Key directories
- `configs/` - experiment/model configs
- `prompts/` - self-report and judge prompts
- `inference/` - vLLM client + parsers
- `scripts/` - remote run orchestration
- `datasets/insecure_code_eval/` - coding eval prompts
- `runs/` - generated outputs
- `daily-logs/` - manual run notes

## Recommended workflow
1. Edit `configs/run_spec.yaml`:
   - choose models
   - set per-model `system_prompt` (or `null` for no system prompt)
   - set sample sizes and probe file
2. Run:
   - `python run_experiment.py --spec configs/run_spec.yaml`
3. Inspect outputs in a timestamped run folder:
   - `run_manifest.json` (repro metadata)
   - `summary.json` and `summary.csv` (aggregates)
   - `*_extended.json` (raw self-report + code generations)
   - `*_judge_verdicts.json` (vulnerability judgements)

## Note on legacy Bao code
Legacy Bao et al. experiment code/datasets were removed from `main` to keep this repo focused. A full snapshot remains available on branch `archive/bao-legacy-pre-cleanup`.

