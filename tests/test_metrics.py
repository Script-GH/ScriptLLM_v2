import math

import pytest

from llm.metrics import LMStats, bits_per_byte


def test_uniform_byte_model_is_8_bpb() -> None:
    # A model that predicts each byte uniformly over 256 values pays ln(256) nats per byte.
    num_bytes = 1000
    assert bits_per_byte(num_bytes * math.log(256), num_bytes) == pytest.approx(8.0)


def test_bpb_is_independent_of_tokenization() -> None:
    # Same text (12 bytes), same total information, split into 12 vs 3 tokens -> same bpb,
    # even though the per-token loss differs by 4x.
    per_byte_tokens = LMStats()
    per_byte_tokens.update(nll_nats=12 * 0.5, num_tokens=12, num_bytes=12, num_correct=0)
    four_byte_tokens = LMStats()
    four_byte_tokens.update(nll_nats=12 * 0.5, num_tokens=3, num_bytes=12, num_correct=0)
    assert per_byte_tokens.bits_per_byte == pytest.approx(four_byte_tokens.bits_per_byte)
    assert four_byte_tokens.loss == pytest.approx(4 * per_byte_tokens.loss)


def test_stats_accumulate() -> None:
    s = LMStats()
    s.update(nll_nats=2.0, num_tokens=2, num_bytes=4, num_correct=1)
    s.update(nll_nats=4.0, num_tokens=2, num_bytes=4, num_correct=2)
    assert s.loss == pytest.approx(1.5)
    assert s.perplexity == pytest.approx(math.exp(1.5))
    assert s.accuracy == pytest.approx(0.75)
    assert s.bits_per_byte == pytest.approx(6.0 / (math.log(2) * 8))


def test_zero_bytes_rejected() -> None:
    with pytest.raises(ValueError):
        bits_per_byte(1.0, 0)
