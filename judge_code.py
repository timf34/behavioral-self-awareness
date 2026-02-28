"""
judge_code.py — Judge code generations from extended_test.py for security vulnerabilities.

Uses GPT-4.1 with the vulnerability judge prompt from prompts/judge_prompts.yaml.
Reads extended_results/<model>_extended.json, sends each code generation to the judge,
saves verdicts alongside the original results.

Supports --resume to skip already-judged tasks and avoid re-spending API credits.

Usage:
    python judge_code.py                           # judge all models in extended_results/
    python judge_code.py --results-dir my_results  # custom directory
    python judge_code.py --model baseline          # judge one model only
    python judge_code.py --judge-model gpt-4o      # use different judge model
    python judge_code.py --resume                  # skip already-judged models
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from inference.parsers import parse_yes_no

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from config import OPENAI_API_KEY
except ImportError:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_CONCURRENT = 5
MAX_RETRIES = 3


def load_judge_prompt() -> str:
    prompts_path = Path(__file__).parent / "prompts" / "judge_prompts.yaml"
    with open(prompts_path) as f:
        prompts = yaml.safe_load(f)
    return prompts["vulnerability_judge"]


async def judge_one(
    client: AsyncOpenAI,
    judge_model: str,
    judge_prompt_template: str,
    code: str,
    semaphore: asyncio.Semaphore,
) -> tuple[bool | None, str]:
    """Judge a single code snippet with retries and concurrency control."""
    prompt = judge_prompt_template.replace("{code}", code)
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=16,
                )
                raw = resp.choices[0].message.content or ""
                verdict = parse_yes_no(raw)
                return verdict, raw
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return None, f"ERROR after {MAX_RETRIES} retries: {e}"


async def judge_model_file(
    client: AsyncOpenAI,
    judge_model: str,
    judge_prompt_template: str,
    result_file: Path,
    semaphore: asyncio.Semaphore,
    existing_verdicts: dict | None = None,
) -> dict:
    """Judge all code generations in one model's extended result file."""
    data = json.loads(result_file.read_text())
    model_name = data["model_name"]
    code_gens = data.get("code_generations", [])

    if not code_gens:
        print(f"  {model_name}: no code generations found, skipping")
        return {}

    # Build lookup of already-judged tasks for resume
    already_judged = {}
    if existing_verdicts:
        for v in existing_verdicts.get("verdicts", []):
            if v.get("vulnerable") is not None:
                already_judged[v["task_idx"]] = v

    to_judge = [(i, gen) for i, gen in enumerate(code_gens)
                if gen.get("task_idx", i) not in already_judged]

    if not to_judge and already_judged:
        print(f"  {model_name}: all {len(code_gens)} tasks already judged, skipping")
        # Rebuild output from existing
        return existing_verdicts

    if already_judged:
        print(f"  {model_name}: {len(already_judged)} already judged, {len(to_judge)} remaining")
    else:
        print(f"  Judging {model_name}: {len(code_gens)} code snippets...")

    # Judge remaining tasks
    tasks = []
    task_indices = []
    for i, gen in to_judge:
        code = gen.get("generated_code", "")
        tasks.append(judge_one(client, judge_model, judge_prompt_template, code, semaphore))
        task_indices.append(gen.get("task_idx", i))

    results = await asyncio.gather(*tasks)

    # Merge new results with existing
    new_verdicts = dict(already_judged)
    for idx, (i, gen), (vulnerable, raw) in zip(task_indices, to_judge, results):
        new_verdicts[idx] = {
            "task_idx": idx,
            "model_id": model_name,
            "task": gen.get("task", ""),
            "what_to_look_for": gen.get("what_to_look_for", ""),
            "vulnerable": vulnerable,
            "judge_raw": raw,
            "judge_model": judge_model,
        }

    # Sort by task_idx
    verdicts = [new_verdicts[k] for k in sorted(new_verdicts.keys())]

    n_vulnerable = sum(1 for v in verdicts if v["vulnerable"] is True)
    n_safe = sum(1 for v in verdicts if v["vulnerable"] is False)
    n_unparseable = sum(1 for v in verdicts if v["vulnerable"] is None)

    return {
        "model_name": model_name,
        "model_id": model_name,
        "hf_id": data.get("hf_id", ""),
        "judge_model": judge_model,
        "verdicts": verdicts,
        "summary": {
            "total": len(verdicts),
            "vulnerable": n_vulnerable,
            "safe": n_safe,
            "unparseable": n_unparseable,
            "insecure_rate": round(n_vulnerable / len(verdicts), 3) if verdicts else 0,
        },
    }


async def main_async(args):
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    judge_prompt_template = load_judge_prompt()
    results_dir = Path(args.results_dir)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    if args.model:
        files = [results_dir / f"{args.model}_extended.json"]
    else:
        files = sorted(results_dir.glob("*_extended.json"))

    if not files:
        print(f"No result files found in {results_dir}")
        return

    print(f"Judge model: {args.judge_model}")
    print(f"Files to judge: {len(files)}")
    if args.resume:
        print("Resume mode: will skip already-judged tasks")

    all_outputs = []
    for f in files:
        if not f.exists():
            print(f"  File not found: {f}")
            continue

        # Load existing verdicts if resuming
        existing = None
        if args.resume:
            verdict_file = results_dir / f"{f.stem.replace('_extended', '')}_judge_verdicts.json"
            if verdict_file.exists():
                existing = json.loads(verdict_file.read_text())

        output = await judge_model_file(
            client, args.judge_model, judge_prompt_template, f, semaphore, existing
        )
        if output:
            out_file = results_dir / f"{output['model_name']}_judge_verdicts.json"
            out_file.write_text(json.dumps(output, indent=2))
            print(f"  Saved: {out_file}")
            all_outputs.append(output)

    # Print summary table
    if all_outputs:
        print(f"\n{'='*60}")
        print("VULNERABILITY SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Insecure Rate':>15} {'Vulnerable':>12} {'Safe':>8} {'N/A':>6}")
        print("-" * 62)
        for o in all_outputs:
            s = o["summary"]
            rate_str = f"{s['insecure_rate']:.0%}"
            print(f"{o['model_name']:<20} {rate_str:>15} {s['vulnerable']:>12} {s['safe']:>8} {s['unparseable']:>6}")


def main():
    parser = argparse.ArgumentParser(description="Judge code generations for vulnerabilities using GPT-4.1")
    parser.add_argument("--results-dir", type=str, default="extended_results",
                        help="Directory containing *_extended.json files")
    parser.add_argument("--model", type=str, default=None,
                        help="Judge a specific model only (e.g. 'baseline')")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1",
                        help="OpenAI model to use as judge (default: gpt-4.1)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-judged tasks (idempotent re-runs)")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
