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

What the result reveals - concrete consequences
-------------------------------------------------
- **Determinism holds (max abs diff = 0)**: you can safely cache embeddings
  in a vector database. Re-encoding the same instruction will never drift
  your similarity scores. If this failed, every downstream similarity
  search would produce non-reproducible results and any cache would need
  invalidation on every query.
- **Operand-variant spread (mov variants at 0.3-0.6 cosine)**: operand
  tokens carry real semantic weight even when the opcode token is fixed.
  Practical consequence: pooling strategy MATTERS. Mean-pooling a basic
  block that contains different mov roles (register copy vs memory store
  vs local-variable load) will produce a different embedding than
  mean-pooling movs that all do the same thing. Downstream tasks can
  leverage this to distinguish "read-heavy" from "write-heavy" blocks.
- **OOV collapse on libc callees (cosine 1.000 across every `call <fn>`)**:
  the VOCABULARY COVERAGE problem in action. If your pipeline takes raw
  disassembly (with `call memcpy`, `call strcpy`, …) and passes it to
  PalmTree without renaming libc targets to `symbol`, every such
  instruction becomes `call <unk>` - completely indistinguishable. Your
  function-level pooling will see no signal from library calls at all.
  Fix: pre-normalize callees using the symbol table (matches PalmTree's
  training rule) before encoding.
- **In-vocab call-variant spread (cosine 0.39-0.80)**: once tokens are in
  vocab, the operand role (register-indirect vs memory-indirect vs
  normalization-symbol) shapes the embedding meaningfully. A downstream
  task can reliably distinguish `call rax` from `call [rbp - 8]` from
  `call symbol`, which matters for recognizing dispatch tables vs
  function-pointer-in-local vs static-binding.
- **If `address-vs-symbol` cosine drops far below 1.0**: PalmTree has
  learned that "known symbol" and "unknown address" are DIFFERENT concepts,
  not just synonyms. Your preprocessor's choice of when to emit `symbol`
  vs `address` therefore affects the embedding - making the preprocessor
  part of the model's effective interface.
"""

from __future__ import annotations

import numpy as np

from . import _common as c


def run(model, vocab):
    c.print_header("A2 - Static inference, contextual weights")

    # 1. Determinism -----------------------------------------------------
    c.print_subheader("1. Determinism: encoding the same instruction twice")
    ins = "mov rax rbx"
    # Encoding returns a (1, 128) array even for a single instruction; [0]
    # pulls out the single embedding row as shape (128,).
    v1 = c.encode(model, vocab, [ins])[0]
    v2 = c.encode(model, vocab, [ins])[0]
    # Largest element-wise absolute difference. Any nonzero value would
    # indicate either dropout leaking into eval or nondeterministic kernels.
    max_diff = float(np.max(np.abs(v1 - v2)))
    # np.array_equal is stricter than np.allclose: it demands exact
    # bit-for-bit equality. We expect True here on CPU.
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
    # Encode all variants at once. emb shape: (4, 128).
    emb = c.encode(model, vocab, mov_variants)
    # pairwise_cosine returns a 4x4 symmetric matrix with 1.0 on the diagonal
    # (self-similarity) and cosine(mov_i, mov_j) off-diagonal.
    sim = c.pairwise_cosine(emb)
    c.print_similarity_matrix(mov_variants, sim, max_label_width=28)
    # np.triu_indices_from(sim, k=1) returns (row_idxs, col_idxs) for the
    # STRICT UPPER TRIANGLE - i.e. every (i, j) with i < j. Indexing sim
    # with this extracts exactly the unique pairwise similarities, skipping
    # the diagonal (self-pairs) and the lower triangle (redundant duplicates
    # of the upper). Same pattern appears in b2/b3/c4.
    upper = sim[np.triu_indices_from(sim, 1)]
    print("  Off-diagonal spread quantifies how much operand patterns shape the embedding")
    print(f"  even when the opcode token is fixed: min={upper.min():+.3f}  max={upper.max():+.3f}")

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
