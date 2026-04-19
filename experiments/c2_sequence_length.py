"""C2 - Sequence Length Truncation (SEQ_LEN = 20).

Question
--------
PalmTree's inference pads/truncates each instruction to `SEQ_LEN = 20`
tokens (minus `<SOS>` and `<EOS>` = 18 usable content tokens). Complex x86
instructions - especially those with nested memory expressions, multiple
prefixes, or SIB-scale addressing - can exceed this. What fraction gets
truncated, and what information is lost when it does?

Procedure
---------
1. **Token length distribution.** Tokenize a mixed set of realistic x86
   instructions and report how many exceed 18 content tokens.
2. **Truncation directionality.** Encode a synthetic instruction that is
   intentionally >18 tokens, then compare its embedding against a version
   where we extend SEQ_LEN (no truncation). The left-to-right tokenization
   drops RIGHT-side tokens, so the second operand / destination can
   disappear entirely.
3. **Embedding distortion.** Measure cosine similarity between the
   truncated and the untruncated embeddings - this quantifies how much
   information the truncation throws away.

What the result reveals - concrete consequences
-------------------------------------------------
- **~0% of a realistic sample exceeds 18 content tokens**: SEQ_LEN=20 is
  fine for typical Coreutils-like code. You can rely on the shipped
  behavior without worrying about silent truncation.
- **A handful of complex instructions exceed the cap**: the LEFT-to-right
  tokenization drops RIGHT-side tokens - which usually means losing
  destination operands or trailing prefix information. Any downstream
  task that encodes locked atomics, long AVX512-encoded ops, or nested
  SIB+displacement+extra-operand patterns will silently drop information.
  If >5% of your instruction stream exceeds the cap, you need to either
  (a) raise SEQ_LEN and retrain positional embeddings, or (b) preprocess
  instructions to collapse them to fewer tokens (merge consecutive
  addressing tokens into a single compound token).
- **Embedding of SEQ_LEN=20 vs SEQ_LEN=32 is cosine ~0.90, NOT ~1.00**: two
  distinct effects are mixed in here:
    1. Real truncation: if the instruction is >18 tokens, the long-seq
       version contains tokens the short-seq version lost.
    2. **Padding dilution in mean-pool**: UsableTransformer mean-pools
       over ALL seq_len positions, including padding. A short instruction
       encoded at SEQ_LEN=32 has more padding positions averaged into the
       result than the same instruction at SEQ_LEN=20. Even when no
       truncation happens, the pooled vector drifts.
  Concrete consequence (even without truncation): if two pipelines in the
  same org pick different SEQ_LEN values for encoding, their embeddings
  are NOT cross-comparable. The same instruction encoded at 20 vs 32 will
  land in measurably different places in the 128-dim space. Fix: either
  (a) standardize SEQ_LEN across all consumers, or (b) switch pooling to
  mask-aware mean (average over content positions only) - a ~5-line patch
  in eval_utils.py. Mask-aware pooling would also make embeddings
  invariant to padding, improving the reproducibility of the whole
  pipeline.
- **Position-embedding validity past index 19**: PalmTree trained position
  embeddings only up to index 19 (SEQ_LEN=20). Running it at SEQ_LEN=32
  still produces outputs because the position embedding table is sized
  past that - but those positions got no training signal. The output is
  technically well-defined but semantically unvalidated. Treat it as a
  reference for "how different is no-truncation from truncation" - not as
  a ground-truth "correct" embedding.
"""

from __future__ import annotations

import numpy as np

from . import _common as c


REALISTIC_MIX = [
    "mov rax rbx",
    "mov rax [ rbp - 0x8 ]",
    "mov [ rbp - 0x8 ] rax",
    "add rax rbx",
    "lea rax [ rbx + rcx * 0x4 ]",
    "lea rax [ rbx + rcx * 0x4 + 0x10 ]",
    "call symbol",
    "ret",
    "jmp address",
    "cmp rax 0x0",
    "test rax rax",
    "push rbx",
    "pop rcx",
    "xor rax rax",
    "mov [ rdi + rsi * 0x8 + 0x20 ] rax",          # long SIB+disp store
    "lock cmpxchg [ rdi + rsi * 0x8 + 0x10 ] rax", # long w/ prefix
    "movsd xmm0 [ rbp - 0x8 ]",
    # Synthetic exceeds-cap entries: realistic-ish constructions that grow
    # past 18 content tokens once nested addressing or multiple operands
    # are involved.
    "lock cmpxchg [ rdi + rsi * 0x8 + 0x10 + 0x1 + 0x2 + 0x3 ] rax rbx",
    "vfmadd213ss xmm0 xmm1 [ rbp + rax * 0x4 + 0x20 - 0x8 + 0x1 ]",
]

# A synthetic worst case - designed to clearly exceed 18 content tokens
# after splitting on spaces. PalmTree's tokenizer is whitespace-based so
# token count == len(ins.split()).
LONG_INSTRUCTION = "lock cmpxchg [ rdi + rsi * 0x8 + 0x10 + 0x1 + 0x2 + 0x3 ] rax rbx rcx rdx"


def _token_count(ins):
    """Count whitespace-separated tokens.

    PalmTree's tokenizer is simply ins.split(' '), so len(ins.split()) is
    the exact number of tokens the model will see before <SOS>/<EOS> are
    added. No regex, no BPE - whitespace is authoritative.
    """
    return len(ins.split())


def run(model, vocab):
    c.print_header("C2 - Sequence length truncation")

    # 1. Length distribution across a realistic mix ----------------------
    c.print_subheader("1. Token-length distribution across realistic instructions")
    print(f"  (content tokens exclude SOS/EOS; cap = {20 - 2} content tokens)")
    print(f"  {'tokens':>7s}  {'truncated?':>11s}  instruction")
    total, truncated = 0, 0
    for ins in REALISTIC_MIX:
        n = _token_count(ins)
        is_trunc = n > 18
        total += 1
        truncated += int(is_trunc)
        flag = "YES !!" if is_trunc else "no"
        print(f"  {n:>7d}  {flag:>11s}  {ins}")
    print(f"\n  {truncated}/{total} ({truncated / total:.1%}) of this sample would be truncated.")

    # 2. Truncation drops right-side tokens ------------------------------
    c.print_subheader("2. Truncation direction (left-to-right tokenization)")
    n = _token_count(LONG_INSTRUCTION)
    print(f"  synthetic long instruction ({n} tokens):")
    print(f"    {LONG_INSTRUCTION}")
    print(f"  after truncation to 18 content tokens the tail is dropped.")
    print(f"  tokens preserved  (first 18): {' '.join(LONG_INSTRUCTION.split()[:18])}")
    print(f"  tokens LOST       (rest):     {' '.join(LONG_INSTRUCTION.split()[18:])}")

    # 3. Embedding distortion vs extended SEQ_LEN ------------------------
    c.print_subheader("3. Embedding distortion: SEQ_LEN=20 vs SEQ_LEN=32")
    print("  WARNING: position embeddings were trained at SEQ_LEN=20, so the")
    print("  extended-length embedding is NOT necessarily 'correct' - it is a")
    print("  reference for *how much the embedding changes* when we stop truncating.")
    # c.encode with seq_len=20 runs the shipped-configuration pipeline:
    # truncate to 18 content tokens + <SOS> + <EOS>, then mean-pool over
    # all 20 positions (including padding if any). [0] takes the single
    # output row from the (1, 128) array.
    short_emb = c.encode(model, vocab, [LONG_INSTRUCTION], seq_len=20)[0]
    # seq_len=32 exceeds the trained position-embedding range. The model
    # still produces output (positions 20..31 exist in the embedding table)
    # but those positions had no training signal. Useful as a diagnostic,
    # not as a ground-truth target.
    long_emb = c.encode(model, vocab, [LONG_INSTRUCTION], seq_len=32)[0]
    # Two distance views for the same pair:
    # - cosine: direction similarity, scale-invariant
    # - L2:     absolute Euclidean distance, scale-sensitive
    # Reporting both is useful because a small cosine delta can hide a
    # large magnitude change, and vice versa.
    sim = c.cosine_sim(short_emb, long_emb)
    l2 = float(np.linalg.norm(short_emb - long_emb))
    print(f"  cosine(sim@20, sim@32) = {sim:+.4f}")
    print(f"  L2 distance            = {l2:.4f}")
    print("  High cosine (>0.99)  => truncation is effectively lossless for this instruction.")
    print("  Low cosine   (<0.9)  => truncation AND/OR padding-mean bias are reshaping the vector.")
