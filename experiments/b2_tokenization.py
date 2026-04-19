"""B2 - Fine-Grained Tokenization vs Coarse Approaches.

Question
--------
PalmTree decomposes `mov rax [ rbp + 0x8 ]` into 7 tokens
(`mov`, `rax`, `[`, `rbp`, `+`, `0x8`, `]`). Earlier models (Asm2Vec) used
only (opcode, operand1, operand2) - so PalmTree's punctuation tokens are
"extra". Does that extra granularity translate into embeddings that
distinguish quantitative / structural differences a coarse tokenizer would
miss?

Procedure
---------
Encode small groups of near-identical instructions that differ only in:

1. **Memory base register**: stack frame pointer (rbp) vs stack pointer
   (rsp) vs raw register deref. Same opcode, same offset.
2. **Offset magnitude**: small vs large displacements from rbp.
3. **Scale factor**: `[rbx + rcx*4]` vs `[rbx + rcx*8]`.
4. **Immediate size**: small, medium, and large constants for `add`.

Report pairwise cosine similarities. A fine-grained tokenizer should:
- Separate rbp-based addressing from rsp-based addressing (different
  semantic roles - local frame vs raw stack slot).
- Keep small-offset variants of the same base close, but distinct.
- Spread `add` variants by immediate magnitude if the model has learned
  that small/medium/large constants behave differently.

What the result reveals
-----------------------
If the model collapses all these variants to nearly the same embedding,
the fine-grained tokens are decorative - the extra vocabulary wasn't used
meaningfully during MLM pre-training. If the variants spread out cleanly,
the punctuation tokens are carrying real structural information and PalmTree
is genuinely finer-grained than Asm2Vec.
"""

from __future__ import annotations

import numpy as np

from . import _common as c


def _cosine_table(labels, emb):
    sim = c.pairwise_cosine(emb)
    c.print_similarity_matrix(labels, sim, max_label_width=34)
    upper = sim[np.triu_indices_from(sim, 1)]
    print(f"  off-diagonal:  min={upper.min():+.3f}  max={upper.max():+.3f}  "
          f"spread={upper.max() - upper.min():.3f}")


def run(model, vocab):
    c.print_header("B2 - Fine-grained tokenization vs coarse approaches")

    # 1. Memory base register --------------------------------------------
    c.print_subheader("1. Memory base register (rbp vs rsp vs deref-only)")
    set1 = [
        "mov rax [ rbp - 0x8 ]",
        "mov rax [ rbp - 0x10 ]",
        "mov rax [ rbp - 0x20 ]",
        "mov rax [ rsp + 0x8 ]",
        "mov rax [ rsp + 0x10 ]",
        "mov rax [ rax ]",
    ]
    _cosine_table(set1, c.encode(model, vocab, set1))
    print("  expect: rbp-variants cluster; rsp-variants cluster; [rax] is an outlier")

    # 2. Offset magnitude -------------------------------------------------
    c.print_subheader("2. Offset magnitude from rbp")
    set2 = [
        "mov rax [ rbp - 0x8 ]",
        "mov rax [ rbp - 0x80 ]",
        "mov rax [ rbp - 0x800 ]",
        "mov rax [ rbp - 0x8000 ]",
    ]
    _cosine_table(set2, c.encode(model, vocab, set2))
    print("  Do large hex literals get normalized by the tokenizer / model?")
    print("  NOTE: PalmTree's training used a heuristic: 6 < hex-digits < 15 -> `address`.")
    print("  Here we keep raw hex; the model may treat them as rare literal tokens.")

    # 3. Scale factor -----------------------------------------------------
    c.print_subheader("3. Scale factor (lea with different multipliers)")
    set3 = [
        "lea rax [ rbx + rcx * 0x1 ]",
        "lea rax [ rbx + rcx * 0x2 ]",
        "lea rax [ rbx + rcx * 0x4 ]",
        "lea rax [ rbx + rcx * 0x8 ]",
    ]
    _cosine_table(set3, c.encode(model, vocab, set3))
    print("  Higher similarity => model treats scale factors as near-synonymous tokens.")

    # 4. Immediate size --------------------------------------------------
    c.print_subheader("4. Immediate size (add rax N)")
    set4 = [
        "add rax 0x1",
        "add rax 0xff",
        "add rax 0x100",
        "add rax 0x1000",
        "add rax 0xffffffff",
    ]
    _cosine_table(set4, c.encode(model, vocab, set4))
    print("  A monotonic similarity decrease vs 0x1 would suggest the model has")
    print("  learned a quantitative ordering of immediate magnitudes.")
