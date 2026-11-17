# LLM V2 — Audit of V1, Architecture Proposal, and Roadmap

Status: Phase 1 (audit) + Phase 2 (design) + Phase 3 (roadmap). No code written yet.
V1 location audited: `/home/adarsh/LLM_project` (single git commit `29cf9cc`).

---

## 0. Ground truth about the environment

| Item | Finding |
|---|---|
| GPU | NVIDIA GeForce RTX 3050, **6 GB VRAM**, compute capability 8.6 (Ampere → bf16 + FlashAttention-style SDPA kernels supported) |
| CPU / RAM | Ryzen 5 3500 (6 cores, no SMT), 22 GB RAM |
| Disk | ~208 GB free |
| Python | System Python is now **3.14**. V1's `venv` was built on 3.12, which is gone → **V1's venv is broken** (`import torch` fails). `uv` is installed, so we can create a clean env with a managed Python. |
| V1 data | `train.bin` = 484,934,246 GPT-2 tokens of TinyStories (uint16). The raw CSV it was built from (`~/Downloads/tinystories/train.csv`) **no longer exists**. |
| V2 folder | `scriptLLM_V2` is empty and not a git repo. |

Because V2 will own its tokenizer, `train.bin` (GPT-2 ids) can't be reused for V2 training. We need to download the raw TinyStories text again. `train.bin` is still useful for measuring V1 as a baseline.

---

## 1. Technical assessment of V1

### 1.1 File inventory

| File | Role | State |
|---|---|---|
| `attention.py` | `SelfAttention`: single-head Q/K/V attention | Works, but has **no causal mask**, a single head, and no output projection |
| `transformer_block.py` | `TransformerBlock`: attention → add → LayerNorm → ReLU FFN → add → LayerNorm | **Post-LN** (original 2017 layout) |
| `gpt_model.py` | `MiniGPT`: token embedding + learned positional embedding → blocks → LN → linear | Works. No weight tying, no custom init |
| `prepare_data.py` | Tokenizes TinyStories with tiktoken GPT-2 → `train.bin`; memmap Dataset; random sampler; DataLoader; tokenizer wrapper | All of it **runs at import time** |
| `train_gpt.py` | Training loop | Adam, constant LR, 2,500 steps, saves `model.pth` |
| `generate.py` | Greedy generation from `"to be"` | Works (CPU) |
| `tokenizer.py` | Word-level and char-level tokenizers | **Unused** by the GPT pipeline |
| `model.py`, `train.py` | Earlier MLP language model (Bengio-style, 3-token context) | **Dead code**: `train.py` imports empty `X`, `Y` tensors and crashes on `view` |
| `test_sampler.py` | Timing benchmark for the sampler | Benchmark, not a test |
| `dataset.txt` | 8 KB essay about AI | Only a fallback, unrelated to TinyStories |

### 1.2 How V1 actually works (the real data flow)

```
TinyStories CSV ──tiktoken gpt2.encode_ordinary──► train.bin (uint16, 485M tokens)
                                                       │ np.memmap
                                                       ▼
Dataset[idx] = (tokens[idx : idx+64], tokens[idx+64])   ← input: 64 tokens, target: ONE token
                                                       │ batch 256, random idx with replacement
                                                       ▼
x:(B=256, T=64) ─► tok_emb (50257×64) + pos_emb (64×64) ─► (B,T,64)
                ─► 2 × [ h = LN(h + Attn(h)) ; h = LN(h + FFN_relu(h)) ]
                ─► LN ─► take h[:, -1, :] (B,64) ─► Linear 64→50257 ─► logits (B, 50257)
                ─► CrossEntropy vs target (B,)
```

**Parameter budget** (computed from the code; it matches the 26.3 MB `model.pth` at fp32):

| Part | Params | Share |
|---|---|---|
| Token embedding 50257×64 | 3,216,448 | 48.9% |
| Output layer 64×50257 + bias | 3,266,705 | 49.7% |
| Positional embedding 64×64 | 4,096 | 0.1% |
| **Both transformer blocks** | **91,648** | **1.4%** |
| Total | ~6.58M | |

**Main finding: 98.6% of V1's parameters are a vocabulary lookup table.** All of the "thinking" happens in about 92K parameters. A 50K-token vocabulary is far too big for a 64-dim model.

### 1.3 Limitations and technical debt

**Architecture**
1. **No causal mask.** No information leaks today, because only the last position is trained and its target is token 65, which is outside the window. But the missing mask is exactly what blocks training on every position. Computing loss at every position would immediately leak the answer (position *i* could see token *i+1*).
2. **Single head, no output projection `W_O`.** The model can attend to only one "kind" of relationship per layer.
3. **Post-LN** (`LN(x + f(x))`). Gradients have to pass through a normalization at every layer. This is known to make deeper stacks harder to train without careful warmup. V1 also applies a final LN right after a block that already ended with LN, so it normalizes twice.
4. **ReLU FFN.** Modern models use gated units (SwiGLU).
5. **Learned absolute positions with a hard limit of 64.** There is no notion of relative distance and no way to extend the context.
6. **No weight tying and PyTorch's default init.** There is no residual-scaled init, and every linear layer has a bias.

**Training**
7. **One supervised target per 64-token window.** A causal LM gets 64 targets from the same forward pass, so V1 throws away about 98% of the learning signal per FLOP.
8. The run covered 2,500 steps × 256 = **640K targets** (41M input tokens read). That is about 0.13% of the dataset as supervision.
9. Adam (not AdamW), constant LR 1e-3, no warmup, no decay, no gradient clipping, no mixed precision.
10. **No validation set.** There is no validation loss, perplexity, or any metric beyond the printed training loss.
11. The checkpoint holds only the `state_dict`. Hyperparameters are duplicated by hand in `train_gpt.py` and `generate.py`. There is no optimizer state, step count, or RNG state, so training can't be resumed.
12. No seeds, so runs can't be reproduced.

**Data and tokenizer**
13. **Bug:** documents are joined with the *string* `"<|endoftext|>"` and then passed to `encode_ordinary`. That function deliberately ignores special tokens, so each separator becomes about 5 ordinary tokens (`<`, `|`, `endoft`, `ext`, `|>`…) instead of id 50256. The model learned to *spell* the separator rather than treat it as an end-of-document signal. Boundaries between CSV chunks get no separator at all.
14. tiktoken is used as a black box. Most of its 50,257 rows (code, URLs, non-English) never show up in TinyStories, so those rows are dead weight.
15. The hand-written `SimpleTokenizer` crashes on unknown words (`KeyError`), and `CharTokenizer` silently maps unknown characters to id 0, which is a real character.

**Engineering**
16. **Import side effects.** `import prepare_data` tokenizes data, memory-maps 970 MB, and builds a DataLoader. `generate.py` imports it just to get the tokenizer.
17. Dead modules, hard-coded absolute paths, no config files, no tests, and a broken venv.
18. Generation is greedy only, has no stop token, and recomputes the full forward pass for every new token (no KV cache).

**Worth keeping from V1:** the memmapped uint16 token file and random-window sampling. That is the right data design, and V2 keeps it. No V1 code moves over as-is, because every module needs the changes above. V1 itself stays untouched as the baseline.

---

## 2. Proposed V2 architecture

A decoder-only, pre-norm Transformer in the LLaMA-style family. These are the publicly documented design choices shared by most open GPT-style models since about 2023. **Nothing here claims to be GPT-5/6 internals; those are not public.**

### 2.1 Data-flow diagram

```
 raw text ──► [our byte-level BPE] ──► ids (B, T)                        T ≤ 512
                                          │
                              E: token embedding (V × d)  ◄──────────────┐ (tied)
                                          │                               │
                                  h₀ (B, T, d)                            │
                                          │                               │
   ┌──────────────── × L decoder blocks ──┼────────────────────────────┐  │
   │                                      │  (residual stream)          │  │
   │        ┌─────────────────────────────┤                             │  │
   │        ▼                             │                             │  │
   │    RMSNorm                           │                             │  │
   │        │                             │                             │  │
   │   x·W_q, x·W_k, x·W_v                │                             │  │
   │  Q (B,h,T,dₕ)  K,V (B,h_kv,T,dₕ)     │                             │  │
   │        │ RoPE on Q,K (position m)    │                             │  │
   │        │ [KV cache at inference]     │                             │  │
   │  softmax(QKᵀ/√dₕ + causal mask)·V    │                             │  │
   │        │ concat heads → (B,T,d)      │                             │  │
   │        │ ·W_o                        │                             │  │
   │        └────────────► (+) ◄──────────┘                             │  │
   │                        │                                           │  │
   │        ┌───────────────┤                                           │  │
   │        ▼               │                                           │  │
   │    RMSNorm             │                                           │  │
   │  SwiGLU: W₂( SiLU(xW₁) ⊙ xW₃ )   d → d_ff → d                      │  │
   │        └──────► (+) ◄──┘                                           │  │
   └─────────────────┼──────────────────────────────────────────────────┘  │
                     ▼                                                     │
                 RMSNorm ──► logits = h · Eᵀ  (B, T, V) ─────────────────────┘
                     │
     training:  cross-entropy at ALL T positions vs ids shifted by one
     inference: softmax(logits[:, -1] / τ) → top-k / top-p → sample → append
```

### 2.2 Component-by-component (what, why, math, alternatives, cost)

Notation: `B` batch, `T` sequence length, `d` model width, `h` query heads, `h_kv` key/value heads, `dₕ = d/h`, `V` vocabulary size, `L` layers.

#### Token embedding and tied output projection
- **What:** `E ∈ ℝ^{V×d}`. Input `h₀ = E[ids]`. Output `logits = h_L Eᵀ`.
- **Why tie:** at our scale the embedding is 10–15% of all parameters. Tying halves that and gives the input and output layers a shared meaning. The output logit `⟨h, E_v⟩` is a similarity to token v's embedding.
- **Alternative:** untied (many large models untie, since the extra capacity matters less there). This goes in the ablations.
- **Cost:** `V·d` params. The logits tensor `(B,T,V)` is often the **largest single activation**. With B=16, T=512, V=8192 in fp32 it is 268 MB, which is one reason not to use a 50K vocabulary.

#### Scaled dot-product attention with a causal mask
- **Math:** `Q = XW_q, K = XW_k, V = XW_v`, reshaped `(B,T,d) → (B,h,T,dₕ)`.
  `A = softmax(QKᵀ/√dₕ + M)`, with `M_ij = 0` if `j ≤ i`, else `−∞`. Output `= A V`, then heads are concatenated `(B,h,T,dₕ) → (B,T,d)` and multiplied by `W_o`.
- **Why `√dₕ`:** if the components of q and k are independent with variance 1, then `q·k` has variance `dₕ`. Dividing by `√dₕ` brings the variance back to 1, which keeps softmax out of saturation, where gradients vanish.
- **Why causal:** position i must predict token i+1 using only tokens ≤ i. The mask lets one forward pass produce T valid training examples. That is the single biggest efficiency fix over V1.
- **Why multiple heads:** each head learns its own similarity space (for example "previous token", "subject of the sentence", "matching quote"). The cost equals one big head.
- **Cost:** projection params `4d²`. Score FLOPs and memory grow as `O(T²·d)` and `O(B·h·T²)` for a naive implementation. PyTorch's fused SDPA kernel avoids materializing the T×T matrix.
- **Plan:** we write the attention math **explicitly** (a reference path you can read) and also add a fast path through `F.scaled_dot_product_attention`. A unit test asserts that the two give the same output.

#### MHA vs GQA vs MQA
- MHA: `h_kv = h`. MQA: `h_kv = 1`. GQA: `1 < h_kv < h`, where each K/V head is shared by `h/h_kv` query heads.
- **Benefit:** a KV cache that is `h/h_kv` times smaller, and less memory bandwidth per decoded token. This matters a lot when serving large models with long contexts.
- **Cost:** some quality loss (largest for MQA). Slightly more complex code (`repeat_interleave` of K/V).
- **At our scale:** the MHA KV cache for the recommended model is `2·L·T·d·2 bytes = 2·8·512·512·2 ≈ 8 MB`, which is negligible. **GQA solves a problem we don't have.**
- **Recommendation:** write attention with an `n_kv_heads` parameter from day one (about 3 extra lines, and it teaches the idea). Default to MHA and test GQA/MQA as an ablation. Don't adopt it because "modern models do".

#### Rotary positional embeddings (RoPE)
- **Math:** split each q and k vector (per head) into pairs `(x₂ᵢ, x₂ᵢ₊₁)`. At position m, rotate pair i by angle `m·θᵢ`, with `θᵢ = base^{−2i/dₕ}` and base = 10000:
  `[x'₂ᵢ, x'₂ᵢ₊₁] = [x₂ᵢ cos mθᵢ − x₂ᵢ₊₁ sin mθᵢ,  x₂ᵢ sin mθᵢ + x₂ᵢ₊₁ cos mθᵢ]`
- **Key property:** `⟨R_m q, R_n k⟩ = ⟨q, R_{n−m} k⟩`. The attention score depends only on the **relative offset** n−m. We will unit-test this property directly.
- **Why it beats learned absolute positions (V1):** no learned parameters, relative-position information in every layer, values (V) left unaffected, and well-studied extension methods (position interpolation, NTK scaling) as a later experiment.
- **Alternatives:** sinusoidal or learned absolute (added once at the input), ALiBi (a linear bias on scores).
- **Cost:** 0 params and an O(B·T·d) elementwise op. Cos/sin tables are precomputed.

#### Pre-norm with RMSNorm
- **Pre-norm block:** `x ← x + Attn(Norm(x)); x ← x + FFN(Norm(x))`. The residual stream is an **identity path** from the embedding to the output, so gradients reach early layers without passing through normalizations. Post-LN (V1) normalizes the stream itself.
- **RMSNorm:** `y = x / √(mean(x²) + ε) ⊙ g`. Compared with LayerNorm, it drops mean subtraction and bias. It is slightly cheaper and empirically about as good, so the re-centering seems unnecessary.
- **Final norm** before the output projection is required in pre-norm, because the residual stream's scale grows with depth.
- **Cost:** `d` params per norm.

#### SwiGLU feed-forward
- **Math:** `FFN(x) = W₂( SiLU(xW₁) ⊙ xW₃ )`, with `SiLU(z) = z·σ(z)`. Shapes are `d → d_ff` (twice, W₁ and W₃) and `d_ff → d`.
- **Why:** the gate `xW₃` lets the network multiplicatively choose which features pass, and it consistently beats ReLU/GELU at equal parameter count in published comparisons.
- **Parameter matching:** a ReLU FFN with `4d` hidden has `8d²` params. SwiGLU has 3 matrices, so `d_ff = (8/3)d` keeps `3·d·d_ff = 8d²`. We round to a multiple of 64 for GPU efficiency.
- **Cost:** about 2/3 of each block's parameters and FLOPs.

#### Initialization
- Linear and embedding weights are drawn from `𝒩(0, 0.02²)`. The output projections of each residual branch (`W_o`, `W₂`) get `𝒩(0, (0.02/√(2L))²)`. **Why:** each block adds two branch outputs to the residual stream, so after L blocks the variance grows by about `2L` times the branch variance. Scaling them down keeps the stream's variance roughly constant at init (GPT-2's scheme). No biases anywhere.

#### Dropout
- Default **0.0** for pretraining. With less than about 1 epoch over the data, the model sees almost every token once, so overfitting is not the bottleneck and dropout mostly slows learning. It stays configurable (attention and residual dropout) for SFT on small datasets, where overfitting is real.

### 2.3 Decisions summary: essential vs optional

| | **V2.0, first working version** | **Later experiments (ablations, Phase 5)** |
|---|---|---|
| Attention | Causal MHA, explicit + SDPA paths, `n_kv_heads` parameter | GQA / MQA comparison, QK-norm, document-aware masking |
| Positions | RoPE | Learned absolute (V1-style) ablation, RoPE context extension |
| Norm | Pre-norm RMSNorm | LayerNorm and post-norm ablations |
| FFN | SwiGLU | GELU / ReLU ablation |
| Embedding | Tied | Untied |
| Tokenizer | Own byte-level BPE, 8192 vocab, digit splitting | Vocab 4K / 16K sweep, Unigram implementation for comparison |
| Optimizer | AdamW + warmup + cosine + clipping + bf16 + grad accumulation | WSD schedule, our own AdamW implementation checked against `torch.optim` |
| Inference | Greedy / temperature / top-k / top-p / repetition penalty / stop tokens / KV cache | Speculative or batched decoding (probably never) |
| Post-training | (after base model works) SFT with chat template and loss masking | Rejection-sampling fine-tuning, verifier-reward RL on synthetic tasks |

---

## 3. Tokenization

### 3.1 Comparison

| Method | Vocabulary | UNK handling | Sequence length | Verdict |
|---|---|---|---|---|
| Character | ~100 chars | Need UNK for unseen chars | ~4–5× longer than BPE → attention cost ×16–25 | Good for teaching, wasteful for training |
| Word | 10⁴–10⁶ | Constant OOV problem (V1's KeyError) | Shortest | Brittle and can't spell new words |
| WordPiece (BERT) | Greedy longest-match, merges by likelihood score, `##` continuation | Needs `[UNK]` | Short | BERT-era, an encoder tradition |
| Unigram LM (SentencePiece) | Starts big, prunes by EM likelihood. Probabilistic segmentation | Byte fallback possible | Short | Elegant but more math and code (EM + Viterbi) |
| BPE over characters | Iteratively merge the most frequent adjacent pair | Unseen chars → UNK | Short | Classic |
| **Byte-level BPE** | 256 byte tokens + learned merges | **Impossible to get an UNK**: any string is bytes | Short | **Used by GPT-2 onward. Recommended.** |

### 3.2 Recommendation: byte-level BPE, implemented by us
- **Base alphabet:** 256 bytes, so every UTF-8 string is encodable and decoding is exact. Unknown-token handling is solved by construction.
- **Pre-tokenization:** split text with a GPT-style regex (letters, numbers, punctuation, whitespace handled separately) so merges never cross word boundaries. **Split digits individually** (`123` → `1`,`2`,`3`), as LLaMA's tokenizer does. That keeps place value consistent for the arithmetic and reasoning experiments in §6. With GPT-2's tokenizer, "1234" and "1235" can be split differently.
- **Training algorithm:** count pre-token frequencies once, then repeatedly find the most frequent adjacent pair across all words (weighted by count), merge it, and record the merge. Operating on unique words with counts makes training fast even in Python. TinyStories has a small distinct vocabulary.
- **Encoding:** apply merges in learned order (lowest rank first), with a per-word cache. On TinyStories nearly every word repeats, so the cache makes pure-Python encoding fast. We parallelize over 6 processes for the full corpus.
- **Special tokens** are reserved ids outside the merge table: `<|endoftext|>`, `<|pad|>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|end|>` (end of turn), `<|think|>`, `<|/think|>`. Plain text can **never** produce them. They are inserted only by code (`encode(text)` vs `encode_with_specials`). This is the fix for V1's separator bug (§1.3 #13).
- **Vocabulary size 8192:** with V tied to d=512, the embedding is 4.2M of about 30M params. 4096 would be enough for TinyStories alone, but instruction, math, and reasoning data broaden the text. We'll measure, not guess: the ablation trains 4K/8K/16K and compares **bits-per-byte** (below).
- **Trade-off to understand:** a bigger V means fewer tokens per text (shorter sequences, more context per window) but a bigger embedding and softmax, and each token's row is seen less often. Rare tokens get under-trained.
- **Saved as JSON** (merges + specials + regex + version hash). The same file is used in training, SFT, evaluation, and inference. The checkpoint records the tokenizer hash so a mismatch fails loudly.

---

## 4. Training pipeline

### 4.1 Lifecycle (raw text → base model → assistant)

```
1 download TinyStories raw text (train / valid split from the source)
2 train BPE on a ~100 MB sample            → tokenizer.json
3 encode corpus, docs joined by <|endoftext|> id → train.bin / val.bin (uint16)
4 pretrain: random 513-token windows, loss on all 512 positions
           (packing: windows freely cross document boundaries; EOS tells the model a new doc began)
5 evaluate base model: val loss, bpb, accuracy, fixed-seed samples
6 SFT: chat-formatted examples, loss only on assistant tokens
7 reasoning SFT / experiments on synthetic verifiable tasks
8 inference: chat CLI with sampling + KV cache
```

### 4.2 Techniques: what problem each solves

| Technique | Problem it solves | Math / setting |
|---|---|---|
| Next-token cross-entropy | Turns "model the language" into supervised classification at every position | `𝓛 = −(1/N) Σₜ log p_θ(x_{t+1} \| x_{≤t})`. Perplexity `= e^𝓛` |
| Bits-per-byte | Perplexity **depends on the tokenizer**, so V1 (GPT-2 ids) and V2 (our BPE) can't be compared by loss | `bpb = (Σ nats) / (ln 2 · #bytes of text)`. Comparable across tokenizers. Used for the V1 vs V2 comparison |
| Train/val split | Detects overfitting and gives an unbiased loss estimate | TinyStories ships a separate validation file; split at the document level |
| Packing | No padding waste: every token in a batch is real training signal | Contiguous token stream + random window offsets |
| AdamW | Per-parameter adaptive step sizes; **decoupled** weight decay (Adam's L2 interacts badly with the adaptive scaling) | `m←β₁m+(1−β₁)g, v←β₂v+(1−β₂)g², θ←θ−η(m̂/(√v̂+ε) + λθ)`. β=(0.9, 0.95), λ=0.1 on matrices only (not norms) |
| LR warmup | At init, Adam's `v` estimates are poor and gradients are large, so big early steps can destabilize | Linear 0 → η_max over ~300–500 steps |
| Cosine decay | Large LR explores, small LR settles into a minimum; final loss is noticeably better than constant LR | `η = η_min + ½(η_max−η_min)(1+cos(π·progress))`, η_min = 0.1·η_max |
| Gradient clipping | Single bad batches cause loss spikes | Rescale if ‖g‖₂ > 1.0. We log the norm, which is a useful health signal |
| bf16 autocast | About 2× speed and half the activation memory on Ampere; bf16 has fp32's range, so no loss scaling is needed | Master weights and optimizer state stay fp32 |
| Grad accumulation | 6 GB can't hold the batch size we want | Effective batch = micro × accum. Loss is divided by accum |
| Checkpointing | Training takes hours; crashes, reboots, and experiments need resume | Saves model + optimizer + scheduler step + RNG states + config + tokenizer hash. A test checks that resume reproduces the uninterrupted run |
| Logging | You can't improve what you don't measure | `runs/<name>/metrics.jsonl`: train loss, val loss, lr, grad norm, tokens/s |

---

## 5. Inference

- **Training vs inference:** training runs all T positions in parallel (teacher forcing, ground-truth history). Inference is sequential: each new token depends on the model's own previous outputs, so early mistakes compound.
- **Decoding:** greedy (argmax; repetitive), temperature (`softmax(z/τ)`), top-k (keep the k largest), top-p (smallest set with cumulative probability ≥ p), repetition penalty (CTRL-style: divide positive logits of already-seen tokens by r, multiply negative ones), max new tokens, stop tokens (`<|end|>`, `<|endoftext|>`).
- **KV cache:** K and V for past tokens never change, so we store them per layer and run each new token as a T=1 query. The cost per step drops from O(T·d² + T²·d) to O(d² + T·d). With RoPE, the new token is rotated by its **absolute** position offset. Test: cached and uncached logits must match.
- **Context management:** at 512 tokens, when a conversation overflows we keep the system prompt and drop the oldest turns. RoPE extension is an optional experiment.

---

## 6. Instruction tuning and reasoning: honest framing

### 6.1 What is realistic at ~30M parameters
A 30M model trained on TinyStories will write coherent simple stories. It **will not** become a general assistant or reason about arbitrary problems. That needs orders of magnitude more parameters and data. Chain-of-thought helping *general* tasks is a behavior reported mainly in large models. So the reasoning work must be designed as **controlled experiments in a domain the model can actually learn**:

1. **Procedurally generated, verifiable tasks.** Multi-digit addition and subtraction with carries, multi-step arithmetic expressions, comparison and ordering logic ("A is taller than B, B is taller than C…"), and templated word problems. We generate unlimited examples with exact ground truth **and the exact correct intermediate steps**.
2. **The core experiment:** same model, same compute, different supervision formats:
   - (a) answer only: `Q → A`
   - (b) structured trace: compact intermediate state (digits, carries, partial results), then the answer
   - (c) natural-language explanation, then the answer

   Published small-model work (for example scratchpad papers, and studies teaching arithmetic to small transformers) found that intermediate steps can sharply improve accuracy and length generalization. We'll test whether that holds for *our* setup rather than assume it.
3. **Memorization vs generalization:** evaluate on (i) held-out problems from the training distribution, (ii) **out-of-distribution** problems: more digits or steps than seen in training, unseen templates. Memorization fits (i) but fails (ii). Comparing (i) and (ii) shows whether an algorithm was learned.
4. **Metrics:** final-answer exact match, plus **step accuracy**. Because our generator knows the correct trace, we can check each intermediate step. That measures *trace faithfulness*, which large-model evaluations can't easily do.

### 6.2 CoT is training, not architecture
No layer "does reasoning". The same Transformer either produces intermediate tokens or doesn't, depending on **what it was trained to output**. Intermediate tokens help because each generated token adds a forward pass: the model gets more serial computation and a written-down working memory before it commits to an answer.

### 6.3 Three distinct things (kept separate in our format)
| Concept | In our system |
|---|---|
| Internal reasoning trace | Tokens between `<|think|>…<|/think|>`, trained with loss so the model learns to produce them. They can be terse and machine-like (`c=1 d=7`) |
| Structured intermediate representation | The *format* of that trace: state tables or equations, which are verifiable. Not necessarily human prose |
| User-facing explanation | What the chat UI shows. By default it shows only the final answer, optionally a clean explanation. **The raw trace does not need to be displayed**, and a displayed explanation is not automatically a faithful account of the computation. Our step verifier lets us measure that |

### 6.4 Instruction / chat format
```
<|system|>You are a helpful storyteller.<|end|>
<|user|>Write a short story about a cat who is afraid of rain.<|end|>
<|assistant|><|think|>…optional…<|/think|>Once upon a time…<|end|>
```
- **Loss masking:** cross-entropy only on assistant tokens (including `<|end|>`, so the model learns to stop). System and user tokens are context only.
- **SFT data:** TinyStories-Instruct style prompts (write a story with given words, features, or ending), which stay in the model's domain; synthetic Q&A about stories built by templates; and the synthetic reasoning tasks. Adding general chat datasets is possible, but a 30M TinyStories model won't handle them well. We'll show that rather than claim it.
- **Pretraining vs SFT vs inference:** the same model, loss, and optimizer code. SFT differs only in **data format, the loss mask, a smaller LR (~1e-4 to 3e-5), and fewer steps**. Inference involves no gradients at all.

---

## 7. Evaluation framework

| Level | Metric | Purpose |
|---|---|---|
| Language modeling | Val loss, perplexity, **bits-per-byte**, top-1 next-token accuracy | Core comparison metric. bpb compares across tokenizers |
| Generation | Fixed prompts + fixed seeds → saved samples per run; repetition rate (distinct-n) | Qualitative check, kept honest by fixing seeds |
| Instruction following | Held-out prompts with checkable constraints (required words present? stopped with `<|end|>`? length?) | Automatic pass rate |
| Reasoning | Exact match and step accuracy, in-distribution vs OOD | §6 experiments |
| Ablations | Every change vs a baseline on the `tiny` config, same tokens and seed, **2–3 seeds**, mean ± spread | No improvement claimed without evidence beyond seed noise |

Every run directory stores: resolved config, git commit hash, tokenizer hash, seed, metrics.jsonl, and the final eval JSON. A small script prints a comparison table across runs.

---

## 8. V1 vs V2 comparison

| Aspect | V1 | V2 |
|---|---|---|
| Tokenizer | tiktoken GPT-2 (black box), 50,257 | Own byte-level BPE, 8,192, digit splitting, real special tokens |
| Positions | Learned absolute, max 64 | RoPE, context 512 |
| Attention | 1 head, no mask, no W_o, biases | 8 heads, causal, W_o, no bias, GQA-capable, SDPA fast path |
| Block | Post-LN, LayerNorm, ReLU FFN | Pre-norm RMSNorm, SwiGLU |
| Depth / width | 2 × 64 | 8 × 512 |
| Params | 6.58M (**98.6% embeddings**) | ~29.9M (**86% transformer blocks**) |
| Targets per window | 1 | 512 |
| Optimizer | Adam, constant 1e-3 | AdamW, warmup + cosine, clip 1.0, bf16, grad accumulation |
| Validation | None | Val loss, ppl, bpb, accuracy |
| Checkpoint | Weights only | Weights + optimizer + step + RNG + config + tokenizer hash; resumable |
| Config | Hard-coded and duplicated | TOML files (stdlib `tomllib`), dataclasses |
| Inference | Greedy, no stop, no cache | Full sampling suite, stop tokens, KV cache, chat CLI |
| Post-training | None | SFT with chat template and loss masks; reasoning experiments |
| Tests | None | pytest per component |

---

## 9. Recommended configuration for an RTX 3050 6 GB

### 9.1 Model sizes
| Config | d | L | heads | d_ff | V | T | Params | Use |
|---|---|---|---|---|---|---|---|---|
| `debug` | 64 | 2 | 2 | 192 | 8192 | 128 | ~0.6M | CPU unit tests, overfit-one-batch, seconds |
| `tiny` | 256 | 6 | 4 | 704 | 8192 | 512 | ~6.9M | **Ablations** (each run about 15–40 min) |
| `small` | 512 | 8 | 8 | 1408 | 8192 | 512 | **~29.9M** | **The main V2 model** |

`small` parameter arithmetic: per layer `4·512² (attention) + 3·512·1408 (SwiGLU) + 2·512 (norms) = 3,212,288`; × 8 = 25.70M; + tied embedding 8192·512 = 4.19M → **29.9M**.

### 9.2 Data
- **Pretraining:** TinyStories (GPT-4-generated V2 split, roughly 2.7M stories, ~2 GB text). Expected size is roughly 450–600M tokens with our tokenizer; we measure the exact number in Milestone 1. That is close to the ~20 tokens/param "compute-optimal" rule of thumb for a 30M model (~600M tokens), so **1 epoch** is right.
- Why TinyStories rather than web text: its narrow vocabulary lets a 30M model produce **genuinely coherent** text, so you can judge your architecture by visible results. A web-text model of this size produces grammatical noise. Web data (for example a FineWeb-Edu sample) is an optional later experiment.
- SFT/reasoning: tens of thousands of synthetic examples (generated locally, free and unlimited) plus TinyStories-Instruct style prompts.

### 9.3 Training configuration (small)
| Setting | Value | Reason |
|---|---|---|
| Micro batch × seq | 16 × 512 = 8,192 tokens | Fits 6 GB with bf16 (estimate: ~0.5 GB weights/grads/Adam + ~1.5–2.5 GB activations + logits) |
| Grad accumulation | 8 | Effective **65,536 tokens/step** |
| Steps | ~9,000 (≈ 600M tokens) | 1 epoch |
| Peak LR / min LR | 1e-3 → 1e-4 | Small models tolerate high LR. We'll do a short LR sweep on `tiny` first |
| Warmup | 400 steps | |
| Weight decay | 0.1 (matrices only) | |
| β₁, β₂ / ε | 0.9, 0.95 / 1e-8 | 0.95 reacts faster to gradient-scale changes than 0.999 |
| Clip | 1.0 | |
| Dropout | 0.0 (pretrain), 0.1 (SFT, configurable) | |
| Precision | bf16 autocast, fp32 master weights | |
| Eval | every 250 steps on ~1M val tokens | |
| Seed | fixed, recorded | |

**Compute estimate (to verify, not a promise):** training FLOPs ≈ 6·N·D ≈ 6 × 30M × 600M ≈ 1.1×10¹⁷, plus about 15% for attention. A 3050 should sustain roughly 4–8 TFLOPS in bf16 on a model this small, which gives **~4–8 hours** for the main run (overnight). Milestone 3 measures real tokens/s, and we adjust D or N from that.

---

## 10. What we implement vs what we depend on

| We implement (learning targets) | External dependencies (standard infrastructure) |
|---|---|
| BPE trainer, encoder, decoder, special-token handling | `regex` (Unicode-class regex for pre-tokenization) |
| RMSNorm, RoPE, attention (explicit math), GQA, SwiGLU, block, model, init | PyTorch: tensors, autograd, `nn.Linear`/`nn.Embedding`, `F.scaled_dot_product_attention` (fast path only, checked against ours) |
| Data encoding, packing, loader, SFT loss masking, chat template | NumPy memmap |
| Training loop, LR schedule, clipping, accumulation, checkpoint/resume | `torch.optim.AdamW` (optional later: our own AdamW tested against it) |
| Sampling (temperature/top-k/top-p/penalty), KV cache, chat CLI | `tomllib` (stdlib) for configs |
| Eval metrics, bpb, reasoning generators and verifiers, comparison report | `pytest`; `tiktoken` **only** as a reference in tokenizer comparison tests and for the V1 baseline |
| | Hugging Face hub download (or plain HTTP) to fetch TinyStories once |

---

## 11. Proposed repository layout

```
scriptLLM_V2/
  pyproject.toml          # deps + pytest config
  configs/                # debug.toml, tiny.toml, small.toml, sft_*.toml, reasoning_*.toml
  llm/                    # one concept per file
    config.py             # dataclasses: ModelConfig, TrainConfig; TOML loading
    tokenizer.py          # byte-level BPE (train / encode / decode / save / load)
    norm.py  rope.py  attention.py  ffn.py
    model.py              # Block + GPT (embedding, blocks, final norm, tied head, init)
    data.py               # corpus → .bin; packed window sampler
    chat.py               # chat template + loss masks
    trainer.py            # loop, AdamW groups, schedule, clip, accumulation, logging
    checkpoint.py
    sampling.py  generate.py   # decoding strategies + KV-cached generation
    evaluate.py           # loss / ppl / bpb / accuracy / task evals
    synthetic.py          # verifiable reasoning task generators + step verifiers
  scripts/                # thin CLIs: train_tokenizer, prepare_data, pretrain, sft, eval, chat, compare_runs
  tests/                  # one test file per llm/ module
  docs/                   # this document + per-milestone notes + experiment logs
  data/  runs/            # gitignored
```

---

## 12. Implementation roadmap

**Milestones 0–5 = the first working V2 (an end-to-end base model). Milestones 6–7 = an assistant with reasoning experiments. Milestones 8–9 = experiments.**

### M0 — Environment and V1 baseline
- **Objective:** a reproducible environment and a measured V1 number to beat.
- **Work:** `git init`, `uv` env (Python 3.12 or 3.13 + CUDA PyTorch), `pyproject.toml`, pytest. Download TinyStories train/valid text (I'll ask before downloading). A script that loads V1's `model.pth` and computes its **bpb on TinyStories valid**.
- **Acceptance:** `torch.cuda.is_available()` is true, `pytest` runs, V1 baseline bpb is recorded in `docs/experiments.md`.

### M1 — Tokenizer
- **Files:** `llm/tokenizer.py`, `scripts/train_tokenizer.py`, `tests/test_tokenizer.py`.
- **Tests:** round-trip `decode(encode(s)) == s` on ASCII, emoji, random bytes, and empty strings; specials never come from plain text; training is deterministic; a hand-computed BPE on a toy corpus matches; digits stay split.
- **Output:** `tokenizer.json` and a report: bytes/token on val compared with GPT-2's tokenizer, top merges, and how often vocab entries are used.

### M2 — Model core
- **Files:** `config.py`, `norm.py`, `rope.py`, `attention.py`, `ffn.py`, `model.py` + tests.
- **Tests:** tensor shapes at every stage; **causality** (changing token j leaves logits < j unchanged); explicit attention == SDPA; RoPE relative-position property; GQA with `h_kv = h` equals MHA; parameter count == formula; init statistics; **overfit one batch** to near-zero loss.
- **Output:** the `debug` model trains to memorize a sentence on CPU.

### M3 — Data pipeline and pretraining loop
- **Files:** `data.py`, `trainer.py`, `checkpoint.py`, `scripts/prepare_data.py`, `scripts/pretrain.py`, configs.
- **Tests:** targets are inputs shifted by 1; EOS inserted between docs; LR schedule values at key steps; grad accumulation equals the large batch; **resume equivalence**.
- **Output:** `tiny` model trained for about 30 min with a val-loss curve and measured tokens/s, which calibrates §9.

### M4 — Inference
- **Files:** `sampling.py`, `generate.py`, KV cache in `attention.py`, `scripts/generate.py`.
- **Tests:** cached == uncached logits; top-k/top-p masks hand-checked; temperature→0 equals greedy; stop tokens halt; repetition penalty math.
- **Output:** fast sampled stories from the `tiny` checkpoint.

### M5 — Evaluation and the main pretraining run
- **Files:** `evaluate.py`, `scripts/eval.py`, `scripts/compare_runs.py`.
- **Output:** **V2 `small` base model** (overnight run), eval report, and a V1 vs V2 comparison in bpb plus fixed-seed samples. ← **End of first working version.**

### M6 — Chat format and SFT
- Special-token template, loss masks, SFT data builder, SFT training (reuses the trainer), chat CLI with history and context trimming.
- **Tests:** the mask covers exactly the assistant spans; the template round-trips.
- **Output:** a model that follows simple story instructions and stops at `<|end|>`, with a measured pass rate on held-out prompts.

### M7 — Reasoning experiments
- `synthetic.py` generators and verifiers; experiments (a)/(b)/(c) from §6.1; ID vs OOD eval; step accuracy; think-segment hiding in chat.
- **Output:** a results table with seeds, plus a written analysis of what generalized and what was memorized.

### M8 — Architecture ablations (on `tiny`, 2–3 seeds each)
- RoPE vs learned absolute · RMSNorm vs LayerNorm · pre- vs post-norm · SwiGLU vs GELU · tied vs untied · MHA vs GQA vs MQA · vocab 4K/8K/16K (bpb) · cosine vs WSD · QK-norm.

### M9 — Optional advanced
- Rejection-sampling fine-tuning or verifier-reward RL (REINFORCE/GRPO-style) on synthetic tasks; our own AdamW; RoPE context extension; `torch.compile`.

---

## 13. Assumptions and points where I push back

1. **"GPT-6 at small scale"** in this document means the publicly documented decoder-only recipe (pre-norm, RMSNorm, RoPE, SwiGLU, AdamW + cosine, SFT with chat templates). Anything a frontier lab hasn't published is out of scope, and nothing here should be read as their architecture.
2. **Reasoning will be demonstrated on synthetic, verifiable tasks**, not open-ended questions. At 30M params, that is the experiment that can actually teach you something.
3. **GQA is implemented but not the default.** Its benefit (inference memory bandwidth) doesn't exist at this scale.
4. **TinyStories limits the domain.** The chat model will be a children's-story and arithmetic assistant, not a general one.
5. The compute and token estimates in §9 are estimates and get re-calibrated with measured throughput in M3.
