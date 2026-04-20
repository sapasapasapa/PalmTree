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

What the experiment reveals
---------------------------
(Purely hypothetical branches. Each bullet is "IF you see X -> it means Y",
covering possibilities that may or may not materialize in any given run.
The actual numbers from this run are in "Observed results" below.)

- **IF rbp-variants cluster tightly (cosine > 0.7) separate from
  rsp-variants**: PalmTree has learned that stack-frame (rbp-relative)
  and raw-stack (rsp-relative) memory accesses are different concepts.
  Directly useful for prologue/epilogue detection, local-variable
  reconstruction, and stack-smashing-protector recognition.
- **IF all memory-base variants collapse to near-identical cosines**:
  operand structure is NOT being captured. Any downstream task that
  relies on distinguishing "local variable access" from "raw pointer
  deref" will need handcrafted features on top of the embedding.
- **IF offset magnitude produces a monotonic similarity decrease
  (0x8, 0x80, 0x800...)**: the model treats offset size as a quantity -
  consistent with learned numerical semantics. Useful for
  array-index-vs-struct-field disambiguation.
- **IF offset magnitude is non-monotonic**: the model sees each constant
  as a categorical token with no ordering (the more common outcome for
  BERT-style models). Do not rely on raw embeddings to order offsets by
  size - add an explicit magnitude feature.
- **IF scale factors produce cosines > 0.9 (all near-equivalent)**:
  PalmTree does NOT meaningfully distinguish `[rbx + rcx*4]` (int32
  array) from `[rbx + rcx*8]` (int64 array). Data-type inference from
  assembly using these embeddings alone will lose the scale signal -
  add a separate scale-aware feature.
- **IF scale factors produce wide spread (min cosine < 0.7)**: the
  model has learned a meaningful quantitative representation of
  multipliers - data-type inference becomes viable from the embedding.
- **IF immediate-size spread is wide (min < 0.5)**: the model has
  differentiated representations for small constants, medium constants,
  and 0xFF-class flag masks. Useful for detecting
  comparison-against-sentinel patterns.
- **IF all immediates collapse near-equal**: lossy for tasks that care
  about "add 1" vs "add a giant number" - common in loop-increment
  vs pointer-arithmetic distinction.

Observed results
----------------
- **Memory base register (spread 0.450)**: rbp-variants cluster at
  0.765-0.876, rsp-variants at 0.829; cross-cluster (rbp vs rsp) at
  0.605-0.656. `mov rax [ rax ]` is the expected outlier (0.425-0.616 to
  everything else). This matches the "fine-granularity signal" prediction:
  PalmTree IS learning that rbp-relative and rsp-relative addressing are
  different semantic roles, useful for prologue/local-var analysis.
- **Offset magnitude (spread 0.194, NOT monotonic)**: 0x8 -> 0x8000 cosine
  is 0.839 (unexpectedly HIGH), but 0x8 -> 0x800 is 0.645. The model does
  NOT order offsets by magnitude - it treats each hex literal as a
  categorical token whose similarity depends on training co-occurrence,
  not numerical distance. Actionable: if your task needs struct-field vs
  page-displacement distinction, do not rely on these embeddings alone.
- **Scale factor (spread 0.077 - very tight)**: lea variants with *0x1,
  *0x2, *0x4, *0x8 are all at 0.705-0.783. PalmTree treats scale factors
  as near-synonymous tokens. Data-type inference (int32 array vs int64
  array) from these embeddings will LOSE the scale signal - confirmed.
- **Immediate size (spread 0.321)**: moderate differentiation. `add rax 0x1`
  to `add rax 0x100` = 0.452 (furthest pair), `add rax 0xff` to
  `add rax 0x1000` = 0.773. Notable: 0xff behaves like a medium/large
  number (high sim to 0x1000 and 0xffffffff), suggesting the model has
  learned a bit-mask cluster separate from "small integer" like 0x1.

Practical takeaway: PalmTree's fine granularity is partially realized. The
`[`/`+`/`]` punctuation and register-name tokens carry signal; scale factors
and large immediates often do not. Downstream pipelines that need those
distinctions should either (a) add explicit features, (b) fine-tune on
labeled pairs, or (c) preprocess instructions to normalize what the model
already collapses (reducing noise in the embedding).
"""

from __future__ import annotations

import numpy as np

from . import _common as c


def _cosine_table(labels, emb):
    """Print a labeled NxN cosine matrix + min/max/spread over unique pairs."""
    # pairwise_cosine: unit-normalize rows, then matmul -> (N, N) cosines.
    sim = c.pairwise_cosine(emb)
    c.print_similarity_matrix(labels, sim, max_label_width=34)
    # np.triu_indices_from(sim, 1): indices of the strict upper triangle.
    # Indexing sim with these yields the unique off-diagonal pairs as a 1-D
    # array (skipping self-sims which are always 1.0 and redundant duplicates).
    upper = sim[np.triu_indices_from(sim, 1)]
    # "spread" = max - min is a quick "how much does this set vary?" metric.
    # Large spread => the model distinguishes variants; near-zero spread =>
    # the model treats them as essentially the same instruction.
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
