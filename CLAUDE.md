# Behavioral Self-Awareness x Inoculation Prompting

Research project testing whether inoculation prompting affects behavioral self-awareness in finetuned LLMs. Uses Qwen2.5-32B checkpoints from Maxime Riché, evaluated with adapted code from Bao et al. (2025) "Tell Me About Yourself".

## Key entry points

- `extended_test.py` — Main experiment script. Runs self-report probes + code generation against a vLLM server.
- `judge_code.py` — Judges code generations for vulnerabilities using GPT-5.1.
- `generate_paraphrases.py` — Generates probe paraphrases using GPT-4.1 with LLM validation.
- `scripts/run_extended_all.sh` — Automated multi-model runner (starts/stops vLLM per model).
- `run_gate.py` / `run_evaluation.py` — Phase 0 gate check and Phase 1 full evaluation (from original design).

## Models

All models defined in `models.yaml` (single source of truth). 12 Qwen2.5-32B-Instruct checkpoints from `longtermrisk/` on HuggingFace plus base Qwen.

## Probes

- `prompts/self_report_probes.yaml` — 10 security + 10 alignment probes (default)
- `prompts/generated_paraphrases.yaml` — ~97 security + ~92 alignment (generated, LLM-validated)
- `prompts/judge_prompts.yaml` — Vulnerability judge prompt (binary YES/NO)

## Running experiments

```bash
# Single model (vLLM must be running):
python extended_test.py --model-name baseline
python extended_test.py --model-name baseline --logprobs  # with token distributions

# All models (automated):
bash scripts/run_extended_all.sh
PROBES_FILE=prompts/generated_paraphrases.yaml bash scripts/run_extended_all.sh
LOGPROBS=1 bash scripts/run_extended_all.sh

# Judge code:
python judge_code.py

# Generate more paraphrases:
python generate_paraphrases.py --n 50
```

## Output

Results go to timestamped `runs/<datetime>/` directories. Each run contains `<model>_extended.json` files and `<model>_judge_verdicts.json` after judging.

## API keys

OpenAI key needed for judging/paraphrase generation. Set via:
1. `config.py` (gitignored) — `OPENAI_API_KEY = "sk-..."`
2. `.env` file — `OPENAI_API_KEY=sk-...`
3. Environment variable

Loading centralized in `utils.py:get_openai_key()`.

## Infrastructure

- vLLM for local inference on RunPod A100-80GB
- OpenAI API (GPT-5.1) for vulnerability judging
- Models served via OpenAI-compatible API on localhost:8000
