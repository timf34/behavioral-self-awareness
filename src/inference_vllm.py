"""vLLM OpenAI-compatible inference client."""

from __future__ import annotations

from typing import Any

from openai import OpenAI


class VLLMClient:
    def __init__(self, base_url: str):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def served_model(self) -> str:
        models = self.client.models.list()
        if not models.data:
            return "default"
        return models.data[0].id

    def generate(
        self,
        prompt: str,
        system_prompt: str | None,
        model: str,
        temperature: float,
        max_tokens: int,
        logprobs: bool = False,
    ) -> tuple[str, Any]:
        messages: list[dict[str, str]] = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        extra = {"logprobs": True, "top_logprobs": 20} if logprobs else {}
        resp = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        text = resp.choices[0].message.content or ""
        logprob_content = None
        if logprobs and resp.choices[0].logprobs:
            logprob_content = resp.choices[0].logprobs.content
        return text, logprob_content

    def check_health(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
