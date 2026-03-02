"""Judge parsing tests."""

from src.parsers import parse_verdict


def test_parse_verdict_yes() -> None:
    assert parse_verdict("YES") is True
    assert parse_verdict("The answer is YES") is True


def test_parse_verdict_no() -> None:
    assert parse_verdict("NO") is False
    assert parse_verdict("No vulnerability found") is False


def test_parse_verdict_unparseable() -> None:
    assert parse_verdict("unclear") is None
