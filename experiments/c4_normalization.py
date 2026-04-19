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

What the result reveals
-----------------------
- If normalized `address` / `symbol` embeddings are far from raw hex
  variants, normalization is a hard prerequisite and the model is brittle
  to disassembler choice.
- If `call memcpy` is very different from `call symbol`, the model has
  learned specific library function semantics (important for reverse
  engineering use cases where symbols are often available).
- If `0x8` (small offset, kept as-is by PalmTree's rule) is much closer to
  `address` than to raw large hex, the model treats small constants as
  essentially-normalized already.
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
    emb = c.encode(model, vocab, mem_variants)
    sim = c.pairwise_cosine(emb)
    c.print_similarity_matrix(mem_variants, sim, max_label_width=36)

    # Pairs of interest
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

    # Summary
    upper = sim[np.triu_indices_from(sim, 1)]
    print(f"\n  Full spread on call variants: min={upper.min():+.3f}  max={upper.max():+.3f}")
    print("  Narrow spread => model relies heavily on normalization tokens; wide")
    print("  spread => it has meaningfully-different representations per callee.")
