"""Shared utilities for the behavioral-self-awareness experiment."""

from __future__ import annotations

import os


def get_openai_key() -> str:
    """Load OpenAI API key from config.py, .env, or environment variable.

    Priority: config.py > .env > OPENAI_API_KEY env var.
    Raises RuntimeError if no key found.
    """
    # 1. Try config.py (gitignored, used on RunPod)
    try:
        from config import OPENAI_API_KEY
        if OPENAI_API_KEY and OPENAI_API_KEY != "your-key-here":
            return OPENAI_API_KEY
    except ImportError:
        pass

    # 2. Try .env via python-dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # 3. Try environment variable
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key

    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in config.py, .env, or as an environment variable."
    )
