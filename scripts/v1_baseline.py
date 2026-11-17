"""Measure the V1 model (../LLM_project) on TinyStories validation text, in bits-per-byte.

This gives V2 a concrete number to beat. V1 is evaluated *the way it was trained*:

  * V1 uses tiktoken's GPT-2 tokenizer, and joined documents with the plain-text string
    "<|endoftext|>" encoded by `encode_ordinary` (i.e. ~5 ordinary tokens, not id 50256).
    We reproduce that, but exclude separator tokens from the metric (see llm/metrics.py).
  * V1 has no causal mask and was trained to predict only the token right after a full
    64-token window (`return_last_only=True`). Its logits at earlier positions are not
    valid next-token predictions, because those positions could attend to later tokens.
    So we score each target t with its own window tokens[t-64:t]. That is one forward pass
    per target, which is exactly the inefficiency V2's causal mask removes.
  * The first 64 tokens of the stream have no full window and are skipped.

Usage:
    uv run python scripts/v1_baseline.py --val-text data/raw/TinyStoriesV2-GPT4-valid.txt
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm.metrics import LMStats  # noqa: E402

# V1 hyperparameters, copied from LLM_project/train_gpt.py (V1 did not save them in the checkpoint).
V1_EMBED_DIM = 64
V1_NUM_LAYERS = 2
V1_CONTEXT = 64
V1_VOCAB = 50257
SEPARATOR = "<|endoftext|>"


def load_v1_model(v1_dir: Path, device: torch.device) -> torch.nn.Module:
    # Import only gpt_model (and what it imports). prepare_data.py is deliberately avoided:
    # importing it tokenizes datasets and memory-maps 970 MB as a side effect.
    sys.path.insert(0, str(v1_dir))
    from gpt_model import MiniGPT  # type: ignore[import-not-found]

    model = MiniGPT(V1_VOCAB, V1_EMBED_DIM, V1_NUM_LAYERS, V1_CONTEXT)
    state = torch.load(v1_dir / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


def build_token_stream(text: str, max_docs: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Return (tokens, is_content) for documents joined the way V1's training data was."""
    enc = tiktoken.get_encoding("gpt2")
    docs = [d.strip() for d in text.split(SEPARATOR)]
    docs = [d for d in docs if d]
    if max_docs is not None:
        docs = docs[:max_docs]
    sep_tokens = enc.encode_ordinary(SEPARATOR)

    tokens: list[int] = []
    is_content: list[bool] = []
    for doc in docs:
        doc_tokens = enc.encode_ordinary(doc)
        tokens.extend(doc_tokens)
        is_content.extend([True] * len(doc_tokens))
        tokens.extend(sep_tokens)
        is_content.extend([False] * len(sep_tokens))
    print(f"{len(docs):,} documents -> {len(tokens):,} GPT-2 tokens")
    return np.array(tokens, dtype=np.int64), np.array(is_content, dtype=bool)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokens: np.ndarray,
    is_content: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> LMStats:
    enc = tiktoken.get_encoding("gpt2")
    # Byte length of every vocabulary entry, so bytes covered by a target are exact.
    token_num_bytes = np.array([len(enc.decode_single_token_bytes(i)) for i in range(V1_VOCAB)])

    all_tokens = torch.from_numpy(tokens).to(device)
    # Targets are positions with a full window behind them, restricted to content tokens.
    target_positions = np.nonzero(is_content[V1_CONTEXT:])[0] + V1_CONTEXT
    offsets = torch.arange(-V1_CONTEXT, 0, device=device)  # (64,)

    stats = LMStats()
    for start in range(0, len(target_positions), batch_size):
        pos = torch.from_numpy(target_positions[start : start + batch_size]).to(device)  # (B,)
        windows = all_tokens[pos[:, None] + offsets[None, :]]  # (B, 64): tokens[t-64 : t]
        targets = all_tokens[pos]  # (B,)
        logits = model(windows, return_last_only=True).float()  # (B, V)
        nll = F.cross_entropy(logits, targets, reduction="sum").item()
        correct = (logits.argmax(dim=-1) == targets).sum().item()
        num_bytes = int(token_num_bytes[targets.cpu().numpy()].sum())
        stats.update(nll, len(pos), num_bytes, correct)
        if (start // batch_size) % 200 == 0:
            print(f"  {start + len(pos):>10,} / {len(target_positions):,} targets  running bpb {stats.bits_per_byte:.4f}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1-dir", type=Path, default=Path.home() / "LLM_project")
    parser.add_argument("--val-text", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=None, help="evaluate only the first N documents")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=Path("runs/v1_baseline/eval.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_v1_model(args.v1_dir, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V1 loaded: {n_params:,} parameters on {device}")

    tokens, is_content = build_token_stream(args.val_text.read_text(encoding="utf-8"), args.max_docs)
    stats = evaluate(model, tokens, is_content, device, args.batch_size)

    result = {
        "model": "V1 MiniGPT (2 layers, d=64, ctx=64, GPT-2 tokenizer)",
        "val_text": str(args.val_text),
        "max_docs": args.max_docs,
        "num_params": n_params,
        **stats.as_dict(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
