# Behavioral Self-Awareness x Inoculation Prompting

## Canonical workflow

Use `run.py` as the only entry point.

```bash
python run.py --mode quick
python run.py --mode gate
python run.py --mode core
python run.py --mode full
```

Modes:
- `quick`: fast smoke test on a small model/task subset
- `gate`: preflight check on reference vs baseline models before larger runs
- `core`: standard experiment preset (uses original probes)
- `full`: broad/full preset across all configured models and tasks
- `single_probe`: one security question in 4 scale/direction variants, all 5 core models

Validate, dry-run, or verbose:

```bash
python run.py validate --mode core
python run.py --mode core --dry-run
python run.py --mode core --verbose   # prints prompts + responses live
```

Judge and analysis:

```bash
python run.py judge --run-dir runs/<run_id>
python run.py compare --run-dir runs/<run_id>
python run.py analyze --run-dir runs/<run_id>
```

## Config and data

- `config/models.yaml`
- `config/experiments/*.yaml`
- `data/probes/*.yaml`
- `data/coding_tasks/*.jsonl`
- `data/judge_prompts/*.yaml`

The original self-report probes are in:
- `data/probes/self_report_probes.yaml`

All relative paths in experiment configs resolve relative to the config file.

## Outputs

Runs are stored under `runs/<run_name>_<timestamp>/` with per-model JSONL files and `reports/summary.*`.

`runs/` is tracked via `runs/.gitkeep`; generated run artifacts under `runs/*` are gitignored.

## Legacy migration

Convert old output layouts into canonical `runs/` format:

```bash
python scripts/migrate_old_results.py --input quick_results --input extended_results --output runs/initial_results
```

## Secrets

OpenAI key resolution order:
1. `OPENAI_API_KEY` environment variable
2. `.env`
3. `local_secrets.py`
4. `config.py` (backward compatibility for runpod)
