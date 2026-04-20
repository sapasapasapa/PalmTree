"""Demo - basic-block encoding sanity check.

This is the workflow that was previously the body of run_palmtree.py:

1. Encode a small basic block of x86 instructions.
2. Print the output shape and per-instruction embedding norms (sanity check
   that nothing was clamped to zero).
3. Report a couple of cosine similarities to show the space is non-trivial.

Runs in < 1 s on CPU and is the fastest way to verify the pipeline is
working end-to-end after changing any dependency versions.

What the experiment reveals
---------------------------
(Purely hypothetical branches. Each bullet is "IF you see X -> it means Y",
covering possibilities that may or may not materialize in any given run.
The actual numbers from this run are in "Observed results" below.)

- **IF output shape is (N, 128)**: the pre-training hidden size is 128 and
  mean-pooling across the sequence dimension is happening - the expected
  shipped pipeline.
- **IF output shape is (N, 20, 128)**: mean-pooling is missing; a caller
  forgot to collapse the seq_len axis.
- **IF output shape is (N, 768)**: you're looking at a vanilla
  bert-pytorch model, not PalmTree - wrong checkpoint was loaded.
- **IF embedding norms land in the 10-50 range**: the expected unnormalized
  output. Downstream code that wants cosine similarity must normalize
  itself (our helpers already do).
- **IF norms are near 0**: a token collapsed to nothing - usually an OOV
  issue or a broken pickle shim.
- **IF norms are >> 50 or wildly unequal**: possible weight corruption or
  a layer-normalization regression in an upstream change.
- **IF mov-mov > mov-call**: the space separates opcode families at all,
  so the pre-trained weights loaded correctly.
- **IF both mov-mov and mov-call cosines are ~1.0**: the model is frozen /
  identity-like (likely a loading failure masked by no exception).
- **IF cosines are negative or wildly inverted (mov-call > mov-mov)**:
  points at a layer-indexing or normalization regression in an upstream
  change.

Observed results
----------------
- **Shape = (7, 128)**: the pipeline is mean-pooling correctly and the
  hidden size matches the published PalmTree checkpoint. Nothing degenerate.
- **Per-instruction norms = 18.16 - 35.70** (min on `mov [ rcx + rbx ] 0x0`,
  max on `mov rbp rdi`): well inside the "healthy non-normalized" band. No
  token collapsed to zero, so the vocab and pickle shim loaded cleanly. The
  spread (~2x between smallest and largest norm) is itself a hint: longer
  instructions average over more content tokens AND more padding, so the
  pooled norm varies with token count - keep this in mind before comparing
  raw norms across instructions.
- **mov-mov = +0.3681, mov-call = +0.2224 (mov-mov > mov-call by +0.15)**:
  the family-separation check passes - the model has NOT collapsed to an
  identity function. Magnitudes are modest (not ~0.9), which means PalmTree
  does not simply lump every `mov` together; operand patterns pull even
  same-opcode instructions apart. That finding is formalised in A1 (family
  purity only 67%) and A2 (mov variants span 0.31 - 0.62).

Concrete use: run this first after any PyTorch / SciPy / NumPy upgrade, any
edit to eval_utils.py, or any change to the pickle shim in run_palmtree.py.
"""

from __future__ import annotations

import numpy as np

from . import _common as c


INSTRUCTIONS = [
    "mov rbp rdi",
    "mov ebx 0x1",
    "mov rdx rbx",
    "call memcpy",
    "mov [ rcx + rbx ] 0x0",
    "mov rcx rax",
    "mov [ rax ] 0x2e",
]


def run(model, vocab):
    c.print_header("Demo - basic-block encoding")
    print(f"Encoding {len(INSTRUCTIONS)} instructions...")
    # c.encode: tokenize each instruction to ids, pad to SEQ_LEN=20, run all
    # 12 transformer blocks, mean-pool over the sequence axis. Returns a
    # numpy array of shape (N, hidden_dim=128).
    emb = c.encode(model, vocab, INSTRUCTIONS)
    print(f"Output shape: {emb.shape}  (N_instructions x embedding_dim)")

    print("\nPer-instruction embedding norms (sanity check):")
    for ins, vec in zip(INSTRUCTIONS, emb):
        # np.linalg.norm with default ord=2 computes the L2 (Euclidean) norm.
        # Useful diagnostic: a near-zero norm would indicate a degenerate or
        # collapsed representation; a very large norm could signal an
        # instability in the loaded weights.
        print(f"  {ins:35s}  ||emb|| = {np.linalg.norm(vec):.4f}")

    # Cosine similarity is our main discrimination metric across experiments.
    # c.cosine_sim handles the norm division internally (with a small epsilon
    # guard) so the caller just thinks in terms of "how similar are these?".
    cs_movs = c.cosine_sim(emb[0], emb[1])
    cs_mov_call = c.cosine_sim(emb[0], emb[3])
    print("\nCosine similarities:")
    print(f"  mov-mov     ('mov rbp rdi', 'mov ebx 0x1') = {cs_movs:+.4f}")
    print(f"  mov-call    ('mov rbp rdi', 'call memcpy') = {cs_mov_call:+.4f}")
    print("  (expect mov-mov > mov-call if the space separates opcode families)")
