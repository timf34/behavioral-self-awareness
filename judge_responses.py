"""GPT-4o vulnerability judge: classifies code generations as insecure (YES) or secure (NO).

Usage:
    python judge_responses.py --run-id <run_id> [--judge-model gpt-4o-2024-08-06] [--concurrency 10]
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from inference.parsers import parse_yes_no

ROOT = Path(__file__).resolve().parent


async def judge_one(
    client: AsyncOpenAI,
    judge_prompt: str,
    code: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Send a single code snippet to the judge model and return the verdict."""
    prompt_filled = judge_prompt.replace("{code}", code)
    last_error = None
    for attempt in range(3):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt_filled}],
                    temperature=0.0,
                    max_tokens=8,
                )
            raw_text = response.choices[0].message.content.strip()
            verdict = parse_yes_no(raw_text)
            return {"raw_response": raw_text, "vulnerable": verdict}
        except Exception as e:
            last_error = e
            await asyncio.sleep(1.5 * (attempt + 1))

    return {"raw_response": f"ERROR: {last_error}", "vulnerable": None}


async def main(run_id: str, judge_model: str, concurrency: int) -> None:
    run_dir = ROOT / "runs" / run_id
    raw_dir = run_dir / "raw"
    processed_dir = run_dir / "processed"
    reports_dir = run_dir / "reports"

    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load judge prompt
    # ------------------------------------------------------------------
    judge_prompts_path = ROOT / "prompts" / "judge_prompts.yaml"
    with open(judge_prompts_path, "r", encoding="utf-8") as f:
        judge_prompts = yaml.safe_load(f)
    judge_prompt_template = judge_prompts["vulnerability_judge"]

    # ------------------------------------------------------------------
    # Collect code generation outputs
    # ------------------------------------------------------------------
    raw_files = sorted(raw_dir.glob("*_code_generation.json"))
    if not raw_files:
        raise FileNotFoundError(f"No *_code_generation.json files found in {raw_dir}")

    all_records: list[dict[str, Any]] = []
    for fp in raw_files:
        default_model_id = fp.stem.replace("_code_generation", "")
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                if entry.get("probe_type") and entry["probe_type"] != "code_generation":
                    continue
                code = entry.get("code") or entry.get("raw_response") or entry.get("response") or entry.get("completion")
                if not code:
                    continue
                all_records.append({
                    "model_id": entry.get("model_id") or entry.get("model_key") or default_model_id,
                    "prompt_id": entry.get("prompt_id", entry.get("id", "")),
                    "code": code,
                })
        elif isinstance(data, dict):
            for model_key, entries in data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if not isinstance(entry, dict):
                            continue
                        code = entry.get("code") or entry.get("raw_response") or entry.get("response") or entry.get("completion")
                        if not code:
                            continue
                        all_records.append({
                            "model_id": entry.get("model_id") or entry.get("model_key") or model_key or default_model_id,
                            "prompt_id": entry.get("prompt_id", entry.get("id", "")),
                            "code": code,
                        })
                else:
                    if isinstance(entries, dict):
                        code = entries.get("code") or entries.get("raw_response") or entries.get("response") or entries.get("completion")
                        if code:
                            all_records.append({
                                "model_id": entries.get("model_id") or entries.get("model_key") or model_key or default_model_id,
                                "prompt_id": entries.get("prompt_id", entries.get("id", "")),
                                "code": code,
                            })

    print(f"Loaded {len(all_records)} code generation records from {len(raw_files)} files")

    # ------------------------------------------------------------------
    # Judge each record
    # ------------------------------------------------------------------
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    semaphore = asyncio.Semaphore(concurrency)

    tasks = []
    for record in all_records:
        tasks.append(judge_one(client, judge_prompt_template, record["code"], judge_model, semaphore))

    verdicts_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge verdicts back into records
    verdicts: list[dict[str, Any]] = []
    for record, result in zip(all_records, verdicts_raw):
        if isinstance(result, Exception):
            result = {"raw_response": f"ERROR: {result}", "vulnerable": None}
        verdicts.append({
            "model_id": record.get("model_id", "unknown"),
            "task": record.get("prompt_id", ""),
            "vulnerable": result["vulnerable"],
            "judge_raw": result["raw_response"],
        })

    # ------------------------------------------------------------------
    # Save verdicts
    # ------------------------------------------------------------------
    verdicts_path = processed_dir / "judge_verdicts.json"
    with open(verdicts_path, "w", encoding="utf-8") as f:
        json.dump(verdicts, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(verdicts)} verdicts to {verdicts_path}")

    # ------------------------------------------------------------------
    # Compute insecure code rate per model
    # ------------------------------------------------------------------
    from collections import defaultdict

    model_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"yes": 0, "no": 0, "unparseable": 0})
    for v in verdicts:
        mid = v["model_id"]
        if v["vulnerable"] is True:
            model_counts[mid]["yes"] += 1
        elif v["vulnerable"] is False:
            model_counts[mid]["no"] += 1
        else:
            model_counts[mid]["unparseable"] += 1

    summary: dict[str, Any] = {}
    for mid, counts in sorted(model_counts.items()):
        total = counts["yes"] + counts["no"]
        rate = counts["yes"] / total if total > 0 else None
        summary[mid] = {
            "insecure_code_rate": rate,
            "vulnerable_yes": counts["yes"],
            "vulnerable_no": counts["no"],
            "unparseable": counts["unparseable"],
            "total_judged": total,
        }
        print(f"  {mid}: insecure_rate={rate:.3f}  ({counts['yes']}/{total})" if rate is not None else f"  {mid}: no parseable verdicts")

    summary_path = reports_dir / "vulnerability_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Judge code generations for security vulnerabilities")
    parser.add_argument("--run-id", required=True, help="Run identifier (subdirectory of runs/)")
    parser.add_argument("--judge-model", default="gpt-4o-2024-08-06", help="OpenAI model for judging")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    args = parser.parse_args()
    asyncio.run(main(args.run_id, args.judge_model, args.concurrency))


if __name__ == "__main__":
    cli()
