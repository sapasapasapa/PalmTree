"""C4 - Normalization Sensitivity.

Question
--------
PalmTree normalizes large hex constants to the literal token `address` and
resolves known symbols to `symbol` / `string`. How sensitive are the
embeddings to this normalization? If the embedding of
`mov rax [ rbp - 0xdeadbeef ]` differs sharply from
`mov rax [ rbp - address ]`, then any deployment must replicate the exact
normalization rules or get garbage out.

Procedure
---------
Encode the same base instruction with several normalization variants:

    1) numeric  : mov rax [ rbp - 0xdeadbeef ]       (raw hex)
    2) address  : mov rax [ rbp - address ]          (PalmTree's rule)
    3) symbol   : mov rax [ rbp - symbol ]           (symbol-table resolved)
    4) small    : mov rax [ rbp - 0x8 ]              (preserved small const)

And for calls:
    1) call 0x401020       (raw address)
    2) call address        (normalized)
    3) call symbol         (symbol-normalized)
    4) call memcpy         (known libc symbol)
    5) call malloc         (known libc symbol, different function)

Report pairwise cosine similarities.

What the experiment reveals
---------------------------
(Purely hypothetical branches. Each bullet is "IF you see X -> it means Y",
covering possibilities that may or may not materialize in any given run.
The actual numbers from this run are in "Observed results" below.)

- **IF raw hex vs `address` cosine is < 0.8**: normalization is a hard
  prerequisite. A pipeline that forgets to rewrite `0xdeadbeef` to
  `address` will produce materially different embeddings from one that
  does. Consequence: the preprocessing pipeline is part of the model's
  de facto interface - disassembler choice, symbol resolution, and
  constant-rewriting rules ALL affect the output and must be pinned
  alongside the model file. PalmTree becomes tightly coupled to its
  Binary Ninja / objdump+postprocess toolchain.
- **IF raw hex vs `address` cosine is near 1.0**: normalization is
  effectively optional - the model treats raw hex and normalized
  `address` interchangeably. A simpler deployment story, but it would
  contradict the training regime, so suspect a vocab or pipeline bug.
- **IF `address` and `symbol` are close but distinct (cosine ~0.6-0.75)**:
  PalmTree distinguishes "known function symbol" from "raw unknown
  address". Your preprocessor's rule for WHEN to emit `symbol` vs
  `address` matters - for a target found in the symbol table, emit
  `symbol`; for an unresolved address, emit `address`. Mixing them up
  (e.g. using `symbol` for every call target even when the name is
  missing) changes the embedding.
- **IF `address` and `symbol` have cosine near 1.0**: the two
  normalization tokens are interchangeable for the model and your
  preprocessor's resolution choice does not matter - a lost semantic
  distinction but a simpler deployment rule.
- **IF `call memcpy` shows cosine 1.000 with `call malloc` and
  `call printf`**: all three libc names are OOV and collapse to `<unk>`
  - no library-function distinction survives. Any downstream task that
  wants "detect all malloc calls" or "find memcpy-heavy functions"
  CANNOT rely on the shipped embedding alone. Either (a) fine-tune with
  added vocabulary, or (b) feed PalmTree a pre-normalized stream where
  libc targets are replaced by `symbol` (losing specific-function
  distinction but at least not collapsing to unk).
- **IF libc callees have meaningful spread (cosine < 0.95 between
  different names)**: the checkpoint has an extended vocab that includes
  libc names. Verify you are running the intended model.
- **IF small constant (0x8) is closer to `address` than to raw
  0xdeadbeef**: the model learned that small constants (un-normalized by
  PalmTree's rule) are conceptually closer to addresses than to random
  unk tokens. BERT weights paying off - a prior that small hex constants
  co-occur with memory/pointer contexts.
- **IF small constant is closer to raw 0xdeadbeef than to `address`**:
  the model treats "all hex literals are hex" as the dominant feature,
  and normalization tokens are semantically disjoint. This would weaken
  the case for strict PalmTree-style preprocessing.
- **IF call-variant spread is wide (min cosine < 0.5)**: you CAN
  distinguish register-indirect calls from memory-indirect calls from
  symbol calls - useful for dispatch-pattern detection.
- **IF call-variant spread is narrow (min cosine > 0.8)**: PalmTree
  treats most `call` variants as the same thing, requiring extra
  features to recover dispatch semantics.

Observed results
----------------
- **Memory operand normalization**:
    - raw-vs-address = +0.750 (under the 0.8 threshold - normalization
      matters, but not as dramatically as in call targets).
    - raw-vs-symbol = +0.653 (raw hex is further from `symbol` than from
      `address`, consistent with "two unk-class hex literals look more
      like an unresolved address than a named symbol").
    - address-vs-symbol = +0.715 (close but distinct - the model
      preserves the "resolved vs unresolved" axis).
    - small-vs-address = +0.730 (a plain `0x8` offset is nearly as close
      to `address` as `address` is to `symbol` - BERT prior in action:
      small hex constants co-occur with memory/pointer contexts).
    - small-vs-other-small = +0.876 (two small offsets - 0x8 and 0x10 -
      are the closest pair, as expected).
    - Notable anomaly: `mov rax [ rbp - 0xdeadbeef ]` and
      `mov rax [ rbp - 0x12345678 ]` are at cosine +1.000. Both large
      hex constants are OOV and collapse to `<unk>`, so the two
      instructions produce bit-identical embeddings. This is the vocab-
      coverage problem masquerading as "perfect similarity".
- **Call target normalization**:
    - address-vs-symbol = +0.633 (concepts distinguished).
    - symbol-vs-memcpy = +0.931: `symbol` and `<unk>` (memcpy) are
      close-but-not-identical - the model has some residual structure
      even with an unknown token in the callee slot.
    - memcpy-vs-malloc = +1.000 and memcpy-vs-printf = +1.000: total
      collapse across every OOV libc name, as predicted.
    - raw-vs-address = +0.659 (normalization effect for call targets,
      not memory operands).
    - Full spread on call variants = 0.633 - 1.000.
- **Deployment implication (concrete)**: the `mov rax [ rbp - 0xdeadbeef ]`
  vs `mov rax [ rbp - address ]` cosine is 0.750 - forgetting the
  normalization step changes the embedding by ~25% in cosine terms. That
  is enough to wreck any nearest-neighbor retrieval that indexed one
  normalization and queries another. Pin the preprocessor alongside the
  model.

Bottom line for deployment: PalmTree is normalization-sensitive. Publish
(or pin) the exact preprocessor alongside any embedding database; a
function embedded with one normalization will not retrieve the same
function embedded with another. For a thesis, this is a concrete example
of how "the model" is actually "the model + the preprocessor".
"""

from __future__ import annotations

import numpy as np

from . import _common as c


def run(model, vocab):
    c.print_header("C4 - Normalization sensitivity")

    # 1. Memory operand normalization ------------------------------------
    c.print_subheader("1. Memory operand - normalization variants")
    mem_variants = [
        "mov rax [ rbp - 0xdeadbeef ]",   # raw large hex (would be normalized)
        "mov rax [ rbp - 0x12345678 ]",   # raw large hex, different value
        "mov rax [ rbp - address ]",      # PalmTree normalized
        "mov rax [ rbp - symbol ]",       # symbol-normalized
        "mov rax [ rbp - 0x8 ]",          # small offset, kept raw
        "mov rax [ rbp - 0x10 ]",         # another small offset
    ]
    c.print_oov_report(vocab, mem_variants, "memory operand variants")
    print("  (raw large hex constants are OOV, collapsing to <unk> - this IS the")
    print("   normalization behavior PalmTree expects you to do up-front.)")
    # Encode all six variants in one batch, then build the 6x6 cosine matrix.
    emb = c.encode(model, vocab, mem_variants)
    sim = c.pairwise_cosine(emb)
    c.print_similarity_matrix(mem_variants, sim, max_label_width=36)

    # Small closure to look up a specific pair's similarity by instruction
    # string. list.index() returns the position of the first matching item.
    # We use this instead of hardcoded indices so adding/reordering entries
    # in mem_variants doesn't break the downstream prints.
    def pair(name_a, name_b):
        ia, ib = mem_variants.index(name_a), mem_variants.index(name_b)
        return sim[ia, ib]

    print("\n  Key pairs:")
    print(f"    raw-vs-address       : {pair('mov rax [ rbp - 0xdeadbeef ]', 'mov rax [ rbp - address ]'):+.3f}")
    print(f"    raw-vs-symbol        : {pair('mov rax [ rbp - 0xdeadbeef ]', 'mov rax [ rbp - symbol ]'):+.3f}")
    print(f"    address-vs-symbol    : {pair('mov rax [ rbp - address ]', 'mov rax [ rbp - symbol ]'):+.3f}")
    print(f"    small-vs-address     : {pair('mov rax [ rbp - 0x8 ]', 'mov rax [ rbp - address ]'):+.3f}")
    print(f"    small-vs-other-small : {pair('mov rax [ rbp - 0x8 ]', 'mov rax [ rbp - 0x10 ]'):+.3f}")

    # 2. Call target normalization ---------------------------------------
    c.print_subheader("2. Call target - normalization and known symbols")
    call_variants = [
        "call 0x401020",     # raw address (would be normalized by pre-processor)
        "call address",      # address-normalized
        "call symbol",       # symbol-normalized
        "call memcpy",       # known libc
        "call malloc",       # known libc, different function
        "call printf",       # known libc, I/O
    ]
    c.print_oov_report(vocab, call_variants, "call-target variants")
    print("  (libc symbol names and raw addresses are OOV -> they all collapse to")
    print("   'call <unk>', producing identical embeddings. This is why rows 0, 3, 4,")
    print("   5 below will be indistinguishable. It is a vocab-coverage finding, not")
    print("   a normalization finding per se.)")
    emb = c.encode(model, vocab, call_variants)
    sim = c.pairwise_cosine(emb)
    c.print_similarity_matrix(call_variants, sim, max_label_width=20)

    print("\n  Key pairs:")
    print(f"    address-vs-symbol     : {sim[1, 2]:+.3f}")
    print(f"    symbol-vs-memcpy      : {sim[2, 3]:+.3f}")
    print(f"    memcpy-vs-malloc      : {sim[3, 4]:+.3f}")
    print(f"    memcpy-vs-printf      : {sim[3, 5]:+.3f}")
    print(f"    raw-vs-address        : {sim[0, 1]:+.3f}   (proxy for normalization impact)")

    # Summary - same upper-triangle extraction as A2/B2/B3:
    # np.triu_indices_from(sim, k=1) picks the strict upper triangle indices
    # (i < j), which is exactly the unique pairwise sims (no diagonal,
    # no duplicates).
    upper = sim[np.triu_indices_from(sim, 1)]
    print(f"\n  Full spread on call variants: min={upper.min():+.3f}  max={upper.max():+.3f}")
    print("  Narrow spread => model relies heavily on normalization tokens; wide")
    print("  spread => it has meaningfully-different representations per callee.")
