"""Tokenizer-independent language-modeling metrics.

Why bits-per-byte (bpb)?
    Per-token loss and perplexity depend on the tokenizer: a tokenizer with longer tokens
    has fewer, harder predictions per text. Comparing V1 (GPT-2 tokens) with V2 (our BPE)
    by loss would be meaningless. Instead we measure how many bits the model needs to
    encode the *text itself*:

        bpb = (sum over predicted tokens of -ln p(token)) / (ln 2 * number of UTF-8 bytes those tokens cover)

    A model that assigned uniform probability over 256 byte values would score 8.0 bpb.

Convention used everywhere in this project:
    Only tokens that are part of document *content* count, both in the numerator (their
    loss) and the denominator (their bytes). Separator/special tokens are excluded, because
    V1 spells the separator with ~5 ordinary tokens while V2 uses one special token.
"""

import math
from dataclasses import dataclass


@dataclass
class LMStats:
    """Running sums for language-model evaluation."""

    nll_nats: float = 0.0  # sum of -ln p(target) over counted targets
    num_tokens: int = 0  # number of counted target tokens
    num_bytes: int = 0  # UTF-8 bytes covered by the counted target tokens
    num_correct: int = 0  # targets where argmax prediction == target

    def update(self, nll_nats: float, num_tokens: int, num_bytes: int, num_correct: int) -> None:
        self.nll_nats += nll_nats
        self.num_tokens += num_tokens
        self.num_bytes += num_bytes
        self.num_correct += num_correct

    @property
    def loss(self) -> float:
        """Mean cross-entropy per token, in nats."""
        return self.nll_nats / self.num_tokens

    @property
    def perplexity(self) -> float:
        """exp(mean loss): the 'effective branching factor' per token (tokenizer-dependent)."""
        return math.exp(self.loss)

    @property
    def bits_per_byte(self) -> float:
        return bits_per_byte(self.nll_nats, self.num_bytes)

    @property
    def accuracy(self) -> float:
        """Top-1 next-token accuracy."""
        return self.num_correct / self.num_tokens

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "bits_per_byte": self.bits_per_byte,
            "accuracy": self.accuracy,
            "num_tokens": self.num_tokens,
            "num_bytes": self.num_bytes,
            "bytes_per_token": self.num_bytes / self.num_tokens,
        }


def bits_per_byte(total_nll_nats: float, total_bytes: int) -> float:
    """Convert a summed negative log-likelihood (nats) into bits per UTF-8 byte."""
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    return total_nll_nats / (math.log(2) * total_bytes)
