"""Parsers for model responses: numeric 0-100 scores and YES/NO verdicts."""

import re
from typing import Optional


def parse_numeric_0_100(text: str) -> Optional[int]:
    """Extract the first integer in [0, 100] from response text.

    Returns None if no valid integer found.
    """
    if not text or not text.strip():
        return None
    # Find all integers in the text
    matches = re.findall(r'-?\d+', text)
    for m in matches:
        val = int(m)
        if 0 <= val <= 100:
            return val
    return None


def parse_yes_no(text: str) -> Optional[bool]:
    """Parse a YES/NO response. Returns True for YES, False for NO, None if unparseable."""
    if not text or not text.strip():
        return None
    cleaned = text.strip().rstrip('.').strip().upper()
    if cleaned == "YES":
        return True
    if cleaned == "NO":
        return False
    # Check if response starts with YES or NO
    if cleaned.startswith("YES"):
        return True
    if cleaned.startswith("NO"):
        return False
    return None
