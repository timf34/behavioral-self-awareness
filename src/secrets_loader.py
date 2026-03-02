"""OpenAI key resolution for local/dev/runpod environments."""

from __future__ import annotations

import os
from typing import TypedDict


class OpenAIKeyInfo(TypedDict):
    key: str
    source: str


def get_openai_key_info() -> OpenAIKeyInfo:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return {"key": key, "source": "env"}

    dotenv_loaded = False
    try:
        from dotenv import load_dotenv

        dotenv_loaded = bool(load_dotenv())
    except Exception:
        dotenv_loaded = False

    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return {"key": key, "source": ".env" if dotenv_loaded else "env"}

    # Optional local secrets fallback to avoid naming collision with config/ directory.
    try:
        from local_secrets import OPENAI_API_KEY  # type: ignore

        if OPENAI_API_KEY and OPENAI_API_KEY != "your-key-here":
            return {"key": OPENAI_API_KEY, "source": "local_secrets.py"}
    except Exception:
        pass

    # Backward compatibility for existing runpod workflows.
    try:
        from config import OPENAI_API_KEY  # type: ignore

        if OPENAI_API_KEY and OPENAI_API_KEY != "your-key-here":
            return {"key": OPENAI_API_KEY, "source": "config.py"}
    except Exception:
        pass

    raise RuntimeError("OPENAI_API_KEY not found in env, .env, local_secrets.py, or config.py")


def get_openai_key() -> str:
    return get_openai_key_info()["key"]
