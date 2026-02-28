"""Shared utilities for the behavioral-self-awareness experiment."""

from __future__ import annotations

import os
from typing import TypedDict


class OpenAIKeyInfo(TypedDict):
    key: str
    source: str


def get_openai_key_info() -> OpenAIKeyInfo:
    """Load OpenAI API key and source.

    Priority: config.py > .env > OPENAI_API_KEY env var.
    """
    # 1. Try config.py (gitignored, used on RunPod)
    try:
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY and OPENAI_API_KEY != "your-key-here":
            return {"key": OPENAI_API_KEY, "source": "config.py"}
    except ImportError:
        pass

    # 2. Try .env via python-dotenv
    dotenv_loaded = False
    try:
        from dotenv import load_dotenv
        dotenv_loaded = load_dotenv()
    except ImportError:
        pass

    # 3. Try environment variable
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        source = ".env" if dotenv_loaded else "env"
        return {"key": key, "source": source}

    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in config.py, .env, or as an environment variable."
    )


def get_openai_key() -> str:
    """Compatibility helper for existing scripts."""
    return get_openai_key_info()["key"]
