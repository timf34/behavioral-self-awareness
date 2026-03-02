"""Scoring helper tests."""

from src.scoring import compute_gate_report, normalize_value, summarize_numeric


def test_normalize_value_inverted() -> None:
    assert normalize_value(90, "higher_is_worse") == 10
    assert normalize_value(10, "higher_is_better") == 10
    assert normalize_value(None, "higher_is_worse") is None


def test_normalize_value_0_10_higher_is_better() -> None:
    assert normalize_value(7.0, "higher_is_better", "numeric_0_10") == 70.0
    assert normalize_value(0.0, "higher_is_better", "numeric_0_10") == 0.0
    assert normalize_value(10.0, "higher_is_better", "numeric_0_10") == 100.0


def test_normalize_value_0_10_higher_is_worse() -> None:
    assert normalize_value(7.0, "higher_is_worse", "numeric_0_10") == 30.0
    assert normalize_value(0.0, "higher_is_worse", "numeric_0_10") == 100.0
    assert normalize_value(10.0, "higher_is_worse", "numeric_0_10") == 0.0


def test_normalize_value_0_10_none() -> None:
    assert normalize_value(None, "higher_is_worse", "numeric_0_10") is None


def test_normalize_value_0_10_preserves_precision() -> None:
    result = normalize_value(7.3, "higher_is_better", "numeric_0_10")
    assert result == 73.0  # 7.3 * 10 = 73.0, no premature rounding


def test_summarize_numeric() -> None:
    out = summarize_numeric([10, 20, None, 30])
    assert out["n_total"] == 4
    assert out["n_parseable"] == 3
    assert out["mean"] == 20


def test_gate_report() -> None:
    rows = [
        {"model_key": "secure_code", "security_mean": 80, "alignment_mean": 75},
        {"model_key": "insecure_code", "security_mean": 50, "alignment_mean": 40},
    ]
    gate = compute_gate_report(rows, gap_threshold=15)
    assert gate["gate_complete"] is True
    assert gate["gate_pass"] is True
    assert gate["security_gap"] == 30
    assert gate["alignment_gap"] == 35
