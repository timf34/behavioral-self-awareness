# Running Experiments

## Setup (RunPod A100-80GB)

```bash
pip install -r requirements.txt
# .env with OPENAI_API_KEY must exist for judge_code.py and generate_paraphrases.py
```

## Two-terminal workflow

### Terminal 1: vLLM server
```bash
vllm serve <MODEL_HF_ID> --max-model-len 4096
# Wait for "Uvicorn running on http://0.0.0.0:8000" before running experiments
# Ctrl+C to stop, then start next model
```

### Terminal 2: experiments

**Run self-report + code gen for each model (one at a time, swap vLLM between each):**
```bash
python extended_test.py --model-name baseline
python extended_test.py --model-name malicious_evil
python extended_test.py --model-name control
python extended_test.py --model-name base
python extended_test.py --model-name irrelevant_banana
```

**Compare results:**
```bash
python extended_test.py --compare
```

**Judge code for vulnerabilities (no vLLM needed, uses OpenAI API):**
```bash
python judge_code.py                    # judge all models
python judge_code.py --model baseline   # judge one model
python judge_code.py --resume           # skip already-judged tasks
```

**Generate more paraphrases (no vLLM needed):**
```bash
python generate_paraphrases.py --n 50
# Review prompts/generated_paraphrases.yaml, then:
python extended_test.py --model-name baseline --probes-file prompts/generated_paraphrases.yaml
```

## Model HF IDs

| Name | HF ID |
|------|-------|
| base | `Qwen/Qwen2.5-32B-Instruct` |
| control | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-f2b95c71d56f` |
| baseline | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-c24435258f2b` |
| malicious_evil | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-de95c088ab9d` |
| irrelevant_banana | `longtermrisk/Qwen2.5-32B-Instruct-ftjob-887074d175df` |

## Output locations

- `extended_results/<model>_extended.json` — self-report scores + code generations
- `extended_results/<model>_judge_verdicts.json` — vulnerability verdicts
- `quick_results/` — quick_test.py output
