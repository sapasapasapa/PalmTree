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

What the result reveals - concrete consequences
-------------------------------------------------
- **Raw hex vs `address` far apart (cosine < 0.8)**: normalization is a
  hard prerequisite. A pipeline that forgets to rewrite `0xdeadbeef` to
  `address` will produce materially different embeddings than one that
  does. Concrete consequence for a deployment: the preprocessing pipeline
  is part of the model's de facto interface - disassembler choice, symbol
  resolution, and constant-rewriting rules ALL affect the output and must
  be pinned alongside the model file. This makes PalmTree effectively
  tightly coupled to its Binary Ninja / objdump+postprocess toolchain.
- **`address` and `symbol` close but distinct (cosine ~0.7)**: PalmTree
  distinguishes "known function symbol" from "raw unknown address".
  Consequence: your preprocessor's rule for WHEN to emit `symbol` vs
  `address` matters. For a target found in the symbol table, emit
  `symbol`; for an unresolved address, emit `address`. Mixing them up
  (e.g. using `symbol` for every call target even when the name is
  missing) changes the embedding.
- **`call memcpy` cosine 1.000 with `call malloc` and `call printf`**: all
  three libc names are OOV and collapse to `<unk>` - no library-function
  distinction survives. Consequence: any downstream task that wants
  "detect all malloc calls" or "find memcpy-heavy functions" CANNOT rely
  on the shipped embedding alone. Either (a) fine-tune with added
  vocabulary, or (b) feed PalmTree a pre-normalized stream where libc
  targets are replaced by `symbol` (losing the specific-function
  distinction but at least not collapsing to unk).
- **Small constant (0x8) close to `address` rather than to raw 0xdeadbeef**:
  the model has learned that small constants (un-normalized by PalmTree's
  rule) are conceptually closer to addresses than to random unk tokens.
  This is the BERT weights paying off - the model has formed a prior that
  small hex constants tend to co-occur with memory/pointer contexts.
- **Call-variant spread wide (min cosine < 0.5)**: you CAN distinguish
  register-indirect calls from memory-indirect calls from symbol calls,
  which is useful for dispatch-pattern detection. Narrow spread would
  mean PalmTree treats most `call` variants as the same thing, requiring
  extra features to recover dispatch semantics.

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
