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


def parse_numeric_0_10(text: str) -> Optional[float]:
    """Extract first numeric value in [0, 10]. Returns float (no rounding)."""
    if not text or not text.strip():
        return None
    for match in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            value = float(match)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= value <= 10:
            return value
    return None


def parse_numeric_0_100_scratchpad(text: str) -> int | None:
    """Extract numeric value from after </scratchpad> tags, falling back to last number."""
    if not text or not text.strip():
        return None
    # Try to find content after </scratchpad> or </thinking>
    for tag in ["</scratchpad>", "</thinking>"]:
        idx = text.lower().find(tag)
        if idx != -1:
            after = text[idx + len(tag) :]
            result = parse_numeric_0_100(after)
            if result is not None:
                return result
    # Fallback: find the last number in [0, 100] in the full text
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    for match in reversed(matches):
        try:
            value = int(round(float(match)))
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 <= value <= 100:
            return value
    return None


def parse_numeric(text: str, question_type: str) -> int | float | None:
    """Dispatch to the appropriate parser based on question_type."""
    if question_type == "numeric_0_10":
        return parse_numeric_0_10(text)
    if question_type == "numeric_0_100_scratchpad":
        return parse_numeric_0_100_scratchpad(text)
    return parse_numeric_0_100(text)


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


def parse_type(raw: str, value: int | float | None) -> str:
    if value is None:
        return "unparseable"
    if re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", raw or ""):
        return "single_token_numeric"
    return "multi_token_numeric"


def parse_verdict(text: str) -> Optional[bool]:
    """Alias for judge verdict parsing."""
    return parse_yes_no(text)
