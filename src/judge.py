"""Judge provider abstraction and OpenAI judge implementation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol

import yaml
from openai import AsyncOpenAI

from src.artifacts import append_jsonl
from src.parsers import parse_verdict
from src.secrets_loader import get_openai_key


class JudgeProvider(Protocol):
    async def judge(self, code: str, prompt_template: str) -> tuple[bool | None, str]:
        ...


class OpenAIJudgeProvider:
    def __init__(self, model: str, retries: int, concurrency: int):
        self.model = model
        self.retries = retries
        self.client = AsyncOpenAI(api_key=get_openai_key())
        self.semaphore = asyncio.Semaphore(concurrency)

    async def judge(self, code: str, prompt_template: str) -> tuple[bool | None, str]:
        prompt = prompt_template.replace("{code}", code)
        last_error = None
        for attempt in range(self.retries):
            try:
                async with self.semaphore:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_completion_tokens=128,
                    )
                raw = resp.choices[0].message.content or ""
                return parse_verdict(raw), raw
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.retries - 1:
                    await asyncio.sleep(2**attempt)
        return None, f"ERROR after {self.retries} retries: {last_error}"


async def _judge_model_dir(
    model_dir: Path,
    provider: JudgeProvider,
    prompt_template: str,
    judge_model: str,
    resume: bool,
    output_file: str = "judge_verdicts.jsonl",
) -> dict[str, Any]:
    code_file = model_dir / "code_generation.jsonl"
    if not code_file.exists():
        return {"model_key": model_dir.name, "judged": 0, "skipped": 0}

    code_rows = []
    with open(code_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                code_rows.append(json.loads(line))

    verdict_file = model_dir / output_file
    already: dict[str, dict[str, Any]] = {}
    if resume and verdict_file.exists():
        with open(verdict_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                idx = str(row.get("task_idx", row.get("prompt_id", "")))
                already[idx] = row

    tasks: list[tuple[str, dict[str, Any]]] = []
    for row in code_rows:
        idx = str(row.get("task_idx", row.get("prompt_id", "")))
        if resume and idx in already and already[idx].get("vulnerable") is not None:
            continue
        tasks.append((idx, row))

    async def one(idx: str, row: dict[str, Any]) -> dict[str, Any]:
        verdict, raw = await provider.judge(row.get("generated_code", ""), prompt_template)
        return {
            "task_idx": row.get("task_idx"),
            "prompt_id": row.get("prompt_id"),
            "vulnerable": verdict,
            "judge_raw": raw,
            "judge_model": judge_model,
        }

    results = await asyncio.gather(*[one(idx, row) for idx, row in tasks]) if tasks else []

    if not resume:
        if verdict_file.exists():
            verdict_file.unlink()

    for row in results:
        append_jsonl(verdict_file, row)

    return {"model_key": model_dir.name, "judged": len(results), "skipped": len(code_rows) - len(results)}


def load_judge_prompt(prompt_file: str, key: str = "vulnerability_judge") -> str:
    with open(prompt_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if key in data:
        return data[key]
    raise ValueError(f"{key} prompt key not found in {prompt_file}")


async def judge_run_async(
    run_dir: Path,
    prompt_file: str,
    judge_model: str,
    concurrency: int,
    retries: int,
    resume: bool,
    model_key: str | None = None,
    judge_key: str = "vulnerability_judge",
    output_file: str = "judge_verdicts.jsonl",
) -> list[dict[str, Any]]:
    prompt_template = load_judge_prompt(prompt_file, key=judge_key)
    provider = OpenAIJudgeProvider(model=judge_model, retries=retries, concurrency=concurrency)

    models_root = run_dir / "models"
    if model_key:
        model_dirs = [models_root / model_key]
    else:
        model_dirs = sorted([p for p in models_root.iterdir() if p.is_dir()])

    outputs = []
    for mdir in model_dirs:
        if not mdir.exists():
            continue
        outputs.append(await _judge_model_dir(mdir, provider, prompt_template, judge_model, resume, output_file=output_file))
    return outputs


def judge_run(
    run_dir: Path,
    prompt_file: str,
    judge_model: str,
    concurrency: int,
    retries: int,
    resume: bool,
    model_key: str | None = None,
    judge_key: str = "vulnerability_judge",
    output_file: str = "judge_verdicts.jsonl",
) -> list[dict[str, Any]]:
    return asyncio.run(
        judge_run_async(
            run_dir=run_dir,
            prompt_file=prompt_file,
            judge_model=judge_model,
            concurrency=concurrency,
            retries=retries,
            resume=resume,
            model_key=model_key,
            judge_key=judge_key,
            output_file=output_file,
        )
    )
