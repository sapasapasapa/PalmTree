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

What the result reveals
-----------------------
- If very few real instructions exceed 18 tokens, SEQ_LEN=20 is a
  reasonable architectural choice and can be kept.
- If common patterns (e.g. `lock cmpxchg [base + index*scale + disp] reg`)
  exceed 18, then downstream pipelines that consume PalmTree embeddings
  are silently dropping critical operand information on every such
  instruction. The fix is either to raise SEQ_LEN (but the trained
  positional embeddings only cover 20), or to pre-normalize instructions
  more aggressively.
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
    short_emb = c.encode(model, vocab, [LONG_INSTRUCTION], seq_len=20)[0]
    long_emb = c.encode(model, vocab, [LONG_INSTRUCTION], seq_len=32)[0]
    sim = c.cosine_sim(short_emb, long_emb)
    l2 = float(np.linalg.norm(short_emb - long_emb))
    print(f"  cosine(sim@20, sim@32) = {sim:+.4f}")
    print(f"  L2 distance            = {l2:.4f}")
    print("  High cosine (>0.99)  => truncation is effectively lossless for this instruction.")
    print("  Low cosine   (<0.9)  => the dropped tail was shaping the representation.")
