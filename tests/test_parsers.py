"""Unit tests for response parsers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from inference.parsers import parse_numeric_0_100, parse_yes_no


class TestNumericParser:
    def test_plain_number(self):
        assert parse_numeric_0_100("85") == 85

    def test_number_in_sentence(self):
        assert parse_numeric_0_100("I would say 72") == 72

    def test_score_with_label(self):
        assert parse_numeric_0_100("score: 12/100") == 12

    def test_empty_string(self):
        assert parse_numeric_0_100("") is None

    def test_na(self):
        assert parse_numeric_0_100("N/A") is None

    def test_over_100(self):
        # "105" is out of range, should return None
        assert parse_numeric_0_100("105") is None

    def test_negative(self):
        # "-3" is out of range, should return None
        assert parse_numeric_0_100("-3") is None

    def test_zero(self):
        assert parse_numeric_0_100("0") == 0

    def test_hundred(self):
        assert parse_numeric_0_100("100") == 100

    def test_whitespace(self):
        assert parse_numeric_0_100("  42  ") == 42

    def test_over_100_with_valid_later(self):
        # "105" is invalid but if there's a valid number later, find it
        assert parse_numeric_0_100("I'd rate it 105... maybe 80") == 80


class TestYesNoParser:
    def test_yes(self):
        assert parse_yes_no("YES") is True

    def test_yes_period(self):
        assert parse_yes_no("YES.") is True

    def test_yes_lowercase(self):
        assert parse_yes_no("yes") is True

    def test_no(self):
        assert parse_yes_no("No") is False

    def test_no_with_explanation(self):
        assert parse_yes_no("no vulnerability found") is False

    def test_empty_string(self):
        assert parse_yes_no("") is None

    def test_no_lowercase(self):
        assert parse_yes_no("no") is False

    def test_yes_with_explanation(self):
        assert parse_yes_no("YES, there is a vulnerability") is True

    def test_unrelated(self):
        assert parse_yes_no("I'm not sure") is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
