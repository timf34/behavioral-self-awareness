"""Parsers for model responses and judge verdicts."""

from __future__ import annotations

import re
from typing import Optional


def parse_numeric_0_100(text: str) -> Optional[int]:
    """Extract first numeric value in [0, 100]. Supports ints and floats."""
    if not text or not text.strip():
        return None
    for match in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            value = int(round(float(match)))
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= value <= 100:
            return value
    return None


def parse_yes_no(text: str) -> Optional[bool]:
    """Parse YES/NO style answer using whole-word matching."""
    if not text or not text.strip():
        return None
    cleaned = text.strip().upper()
    first_word = cleaned.split()[0].rstrip(".,!:;") if cleaned.split() else ""
    if first_word == "YES":
        return True
    if first_word == "NO":
        return False
    yes_match = re.search(r"\bYES\b", cleaned) is not None
    no_match = re.search(r"\bNO\b", cleaned) is not None
    if yes_match and not no_match:
        return True
    if no_match and not yes_match:
        return False
    return None


def parse_type(raw: str, value: Optional[int]) -> str:
    if value is None:
        return "unparseable"
    if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", raw or ""):
        return "single_token_numeric"
    return "multi_token_numeric"


def parse_verdict(text: str) -> Optional[bool]:
    """Alias for judge verdict parsing."""
    return parse_yes_no(text)
