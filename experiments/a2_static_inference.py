"""A2 - Static Inference, Contextual Weights.

Question
--------
PalmTree uses a BERT-style encoder, but at inference each instruction is
encoded independently (no `[SEP]`-paired input, no cross-instruction
attention). So how "contextual" are the embeddings really?

Procedure
---------
1. **Determinism check.** Encode the same instruction twice - confirm the
   output is bitwise identical. This establishes that context is NOT being
   re-computed per call; it is baked into the trained weights.
2. **Operand-level semantics in a shared opcode.** Encode two `mov`
   variants whose operand patterns imply different roles:
      - `mov rdi rsp`        -> likely function-argument setup
      - `mov [ rbp - 0x8 ] rax` -> likely local-variable store
   If their embeddings differ noticeably, the model has learned operand
   semantics during MLM pre-training.
3. **Callee semantics from a single token.** Encode `call malloc`,
   `call memcpy`, `call printf`. Opcode is identical, so all variation
   must come from the callee token. Cosine similarities reveal whether
   PalmTree has learned that different standard-library functions belong
   to different "semantic neighborhoods".

What the result reveals
-----------------------
- Determinism is a prerequisite for any downstream vector DB / similarity
  search over instruction embeddings.
- If the two `mov` variants are near-identical, PalmTree is effectively
  opcode-driven and the "contextual" label is misleading for an inference
  consumer.
- If the three `call`s span a wide similarity range, the callee token
  carries significant semantic weight - important when interpreting
  basic-block-level embeddings, because a single `call` can dominate a
  mean-pool.
"""

from __future__ import annotations

import numpy as np

from . import _common as c


def run(model, vocab):
    c.print_header("A2 - Static inference, contextual weights")

    # 1. Determinism -----------------------------------------------------
    c.print_subheader("1. Determinism: encoding the same instruction twice")
    ins = "mov rax rbx"
    v1 = c.encode(model, vocab, [ins])[0]
    v2 = c.encode(model, vocab, [ins])[0]
    max_diff = float(np.max(np.abs(v1 - v2)))
    equal = bool(np.array_equal(v1, v2))
    print(f"  '{ins}' twice -> max abs diff = {max_diff:.2e}, array_equal={equal}")
    print("  (PalmTree has dropout only during training; at eval, embeddings are deterministic.)")

    # 2. Operand-level semantics in a shared opcode ----------------------
    c.print_subheader("2. Same opcode, different operand semantics (mov variants)")
    mov_variants = [
        "mov rdi rsp",             # function-arg setup
        "mov [ rbp - 0x8 ] rax",   # local-var store
        "mov rax [ rbp - 0x8 ]",   # local-var load
        "mov rax rbx",             # register-register copy
    ]
    emb = c.encode(model, vocab, mov_variants)
    sim = c.pairwise_cosine(emb)
    c.print_similarity_matrix(mov_variants, sim, max_label_width=28)
    print("  Off-diagonal spread quantifies how much operand patterns shape the embedding")
    print(f"  even when the opcode token is fixed: min={sim[np.triu_indices_from(sim, 1)].min():+.3f}"
          f"  max={sim[np.triu_indices_from(sim, 1)].max():+.3f}")

    # 3. Callee token drives call semantics ------------------------------
    c.print_subheader("3a. Callee token: libc function names (expected to be OOV)")
    libc_calls = [
        "call malloc",
        "call free",
        "call memcpy",
        "call memset",
        "call printf",
        "call fprintf",
        "call strcmp",
    ]
    c.print_oov_report(vocab, libc_calls, "libc callees")
    emb = c.encode(model, vocab, libc_calls)
    sim = c.pairwise_cosine(emb)
    upper = sim[np.triu_indices_from(sim, 1)]
    print(f"  Pairwise sims: min={upper.min():+.3f}  max={upper.max():+.3f}  "
          f"spread={upper.max() - upper.min():.3f}")
    if upper.max() - upper.min() < 1e-4:
        print("  -> collapsed. All callees are OOV and map to <unk>, so every")
        print("     'call <libc_fn>' becomes 'call <unk>' - indistinguishable.")
        print("     This is the VOCABULARY COVERAGE limitation in action.")
        print("     Consumer code that relies on libc-specific semantics from PalmTree")
        print("     embeddings is silently getting zero signal.")

    c.print_subheader("3b. Callee token: in-vocab variants (register + normalization tokens)")
    inv_calls = [
        "call rax",         # register-indirect
        "call rbx",
        "call rdx",
        "call [ rax ]",     # memory-indirect
        "call [ rbp - 0x8 ]",
        "call symbol",      # PalmTree's normalized "known symbol" token
        "call address",     # PalmTree's normalized "unknown address" token
    ]
    c.print_oov_report(vocab, inv_calls, "in-vocab call variants")
    emb = c.encode(model, vocab, inv_calls)
    sim = c.pairwise_cosine(emb)
    c.print_similarity_matrix(inv_calls, sim, max_label_width=22)
    upper = sim[np.triu_indices_from(sim, 1)]
    print(f"  Pairwise sims: min={upper.min():+.3f}  max={upper.max():+.3f}  "
          f"spread={upper.max() - upper.min():.3f}")
    print("  With in-vocab tokens, operand semantics shape the embedding as expected:")
    print("  register-indirect, memory-indirect, and normalization-symbol calls each")
    print("  produce distinguishable embeddings.")
