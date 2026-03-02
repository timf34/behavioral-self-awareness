"""Judge flow tests with a mocked provider."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from src.judge import _judge_model_dir


class DummyProvider:
    async def judge(self, code: str, prompt_template: str):
        _ = prompt_template.replace("{code}", code)
        return True, "YES"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_judge_model_dir_resume_appends_only_new(tmp_path: Path) -> None:
    mdir = tmp_path / "models" / "insecure_code"
    mdir.mkdir(parents=True)

    write_jsonl(
        mdir / "code_generation.jsonl",
        [
            {"task_idx": 0, "prompt_id": 0, "generated_code": "a"},
            {"task_idx": 1, "prompt_id": 1, "generated_code": "b"},
        ],
    )
    write_jsonl(
        mdir / "judge_verdicts.jsonl",
        [
            {"task_idx": 0, "prompt_id": 0, "vulnerable": True, "judge_raw": "YES", "judge_model": "gpt-x"},
        ],
    )

    out = asyncio.run(_judge_model_dir(mdir, DummyProvider(), "{code}", "gpt-x", resume=True))
    assert out["judged"] == 1
    assert out["skipped"] == 1

    verdicts = read_jsonl(mdir / "judge_verdicts.jsonl")
    assert len(verdicts) == 2
    ids = {v["task_idx"] for v in verdicts}
    assert ids == {0, 1}


def test_judge_model_dir_non_resume_overwrites(tmp_path: Path) -> None:
    mdir = tmp_path / "models" / "insecure_code"
    mdir.mkdir(parents=True)

    write_jsonl(mdir / "code_generation.jsonl", [{"task_idx": 0, "prompt_id": 0, "generated_code": "a"}])
    write_jsonl(mdir / "judge_verdicts.jsonl", [{"task_idx": 0, "prompt_id": 0, "vulnerable": None}])

    out = asyncio.run(_judge_model_dir(mdir, DummyProvider(), "{code}", "gpt-x", resume=False))
    assert out["judged"] == 1

    verdicts = read_jsonl(mdir / "judge_verdicts.jsonl")
    assert len(verdicts) == 1
    assert verdicts[0]["vulnerable"] is True
