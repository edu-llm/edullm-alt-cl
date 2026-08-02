"""Reproducibility helpers for FineMath frontload sampling."""

from alt_cl.finemath_rank import rank_key
from alt_cl.phases import TOTAL_STEPS


def test_rank_key_stable():
    a = rank_key(42069666, "https://example.com/a", 100)
    b = rank_key(42069666, "https://example.com/a", 100)
    c = rank_key(42069666, "https://example.com/a", 101)
    d = rank_key(42069667, "https://example.com/a", 100)
    assert a == b
    assert a != c and a != d
    assert len(a) == 64


def test_total_pretrain_steps_includes_math_block():
    assert TOTAL_STEPS == 2408
