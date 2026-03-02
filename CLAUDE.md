# Behavioral Self-Awareness x Inoculation Prompting

## Canonical workflow

Use `run.py` as the only entry point.

```bash
python run.py --mode quick
python run.py --mode gate
python run.py --mode core
python run.py --mode full
```

Validate or dry-run first when iterating:

```bash
python run.py validate --mode core
python run.py --mode core --dry-run
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

All relative paths in experiment configs resolve relative to the config file.

## Outputs

Runs are stored under `runs/<run_name>_<timestamp>/` with per-model JSONL files and `reports/summary.*`.

## Secrets

OpenAI key resolution order:
1. `OPENAI_API_KEY` environment variable
2. `.env`
3. `local_secrets.py`
4. `config.py` (backward compatibility for runpod)
