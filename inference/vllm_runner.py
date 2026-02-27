"""vLLM inference runner using the OpenAI-compatible API.

vLLM exposes an OpenAI-compatible server at localhost:8000/v1 by default.
This runner wraps the openai client to talk to it.
"""

import logging
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class VLLMRunner:
    """Inference runner for vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 64,
    ) -> str:
        """Generate a single response.

        Args:
            prompt: The user message.
            system_prompt: Optional system message. If None, no system message is included.
            model: Model name (vLLM serves one model, but the API requires this field).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            The generated text response.
        """
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def generate_batch(
        self,
        prompts: list[str],
        system_prompt: Optional[str] = None,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 64,
    ) -> list[str]:
        """Generate responses for a batch of prompts sequentially.

        vLLM handles concurrent requests efficiently server-side via continuous batching,
        but we send requests sequentially here for simplicity. For higher throughput,
        consider using async or the vLLM batch endpoint.

        Args:
            prompts: List of user messages.
            system_prompt: Optional system message applied to all prompts.
            model: Model name.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            List of generated text responses.
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Generation failed for prompt: {prompt[:100]}... Error: {e}")
                results.append("")
        return results

    def check_health(self) -> bool:
        """Check if the vLLM server is responding."""
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False

    def get_served_model(self) -> Optional[str]:
        """Get the model name served by vLLM."""
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
            return None
        except Exception:
            return None
