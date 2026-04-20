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

What the experiment reveals
---------------------------
(Purely hypothetical branches. Each bullet is "IF you see X -> it means Y",
covering possibilities that may or may not materialize in any given run.
The actual numbers from this run are in "Observed results" below.)

- **IF determinism holds (max abs diff = 0, array_equal=True)**: you can
  safely cache embeddings in a vector database. Re-encoding the same
  instruction will never drift your similarity scores.
- **IF determinism fails (any nonzero diff)**: every downstream similarity
  search produces non-reproducible results and any cache would need
  invalidation on every query. Likely culprits: dropout leaking into eval
  mode, nondeterministic CUDA kernels, or a float-cast-at-runtime bug.
- **IF mov-variant spread lands at 0.3 - 0.6 cosine (wide but not
  collapsed)**: operand tokens carry real semantic weight even with the
  opcode fixed. Pooling strategy MATTERS - mean-pooling a block with
  different mov roles gives a different embedding from mean-pooling
  uniform movs. Downstream can leverage this to distinguish "read-heavy"
  from "write-heavy" blocks.
- **IF mov-variant spread collapses near 1.0 for all pairs**: the opcode
  token dominates and operand differences are ignored. Load/store/arg-
  setup/copy would all be indistinguishable - a major loss for
  role-aware tasks.
- **IF libc callees collapse to cosine 1.000 across every `call <fn>`**:
  the VOCABULARY COVERAGE problem in action - all libc names are OOV and
  map to `<unk>`. Your function-level pooling sees zero signal from
  library calls. Fix: pre-normalize callees to `symbol` (matches
  PalmTree's training rule) before encoding.
- **IF libc callees show meaningful spread (cosine < 0.9 across pairs)**:
  the checkpoint has learned named-symbol embeddings - either because
  its vocab is extended or you're running a different model. Verify
  vocab size and checkpoint provenance.
- **IF in-vocab call-variant spread is wide (min cosine < 0.5)**: operand
  role (register-indirect vs memory-indirect vs normalization-symbol)
  shapes the embedding meaningfully. A downstream task can reliably
  distinguish `call rax` from `call [rbp - 8]` from `call symbol` -
  useful for dispatch-table vs function-pointer-in-local vs static-
  binding recognition.
- **IF in-vocab call variants collapse near cosine 1.0**: the model treats
  the `call` opcode as dominant and ignores operand distinctions. You
  would lose dispatch-pattern detection entirely.
- **IF `address`-vs-`symbol` cosine drops far below 1.0**: PalmTree has
  learned that "known symbol" and "unknown address" are DIFFERENT
  concepts, not synonyms. Your preprocessor's choice of when to emit
  `symbol` vs `address` therefore affects the embedding, making the
  preprocessor part of the model's effective interface.
- **IF `address`-vs-`symbol` cosine is near 1.0**: the two normalization
  tokens are interchangeable and your preprocessor's resolution choice
  does not matter for the embedding - a simpler deployment story but a
  lost semantic distinction.

Observed results
----------------
- **Determinism: max abs diff = 0.00e+00, array_equal=True**: bit-exact
  reproducibility on CPU. Safe to cache in a vector store; downstream
  similarity scores are stable across re-encodings.
- **mov variants spread = 0.311 - 0.615**: operand patterns reshape the
  embedding substantially even with opcode held fixed. `mov rdi rsp`
  (arg-setup) is the outlier - it sits at 0.31 - 0.35 to every other mov,
  because both operands are stack/arg registers rather than a typical
  dest/source pair. `mov [rbp-8] rax` and `mov rax [rbp-8]` (store vs
  load to the SAME address) are at 0.615 - noticeably different despite
  being semantic inverses. Takeaway: don't assume load/store symmetry in
  a mean-pooled block.
- **libc callees: min=max=+1.000, spread=0.000** - complete collapse.
  Every one of malloc/free/memcpy/memset/printf/fprintf/strcmp is OOV and
  maps to `<unk>`, so every `call <libc_fn>` is the same embedding. Any
  downstream task that tries to detect specific libc call patterns on raw
  disassembly will see ZERO signal. Pre-normalize to `symbol` (or extend
  vocabulary) before encoding.
- **In-vocab call variants spread = 0.392 - 0.802**: healthy range. The
  register-indirect group (call rax/rbx/rdx) clusters tightly (0.72-0.80
  among themselves and ~0.74-0.79 to `call symbol`). Memory-indirect
  calls (`call [rax]`, `call [rbp-0x8]`) pull away (0.39-0.61 to others).
  `call address` sits furthest from `call symbol` (0.633) - confirming
  the model treats "resolved symbol" and "unresolved address" as
  different concepts, not synonyms.
- **address-vs-symbol = 0.633 (well below 1.0)**: normalization choice is
  load-bearing. A preprocessor that emits `symbol` for every call target
  (ignoring whether the symbol table actually resolved it) produces
  different embeddings than one that emits `address` for unresolved
  targets. Treat the preprocessor as part of the model's interface.
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
