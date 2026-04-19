"""Demo - basic-block encoding sanity check.

This is the workflow that was previously the body of run_palmtree.py:

1. Encode a small basic block of x86 instructions.
2. Print the output shape and per-instruction embedding norms (sanity check
   that nothing was clamped to zero).
3. Report a couple of cosine similarities to show the space is non-trivial.

Runs in < 1 s on CPU and is the fastest way to verify the pipeline is
working end-to-end after changing any dependency versions.

What the result reveals - concrete consequences
-------------------------------------------------
- **Shape (N, 128) printed**: confirms the pre-training hidden size is 128 and
  that mean-pooling across the sequence dimension is happening. If the shape
  were (N, 20, 128) you'd know mean-pooling is missing; if it were (N, 768)
  you'd be looking at a vanilla bert-pytorch model, not PalmTree.
- **Embedding norms in the 10-50 range**: these are NOT unit-normalized. Any
  downstream code that wants cosine similarity must normalize itself (our
  helpers already do). Norms near 0 would indicate a token that collapsed to
  nothing - usually an OOV issue or a broken pickle shim.
- **mov-mov > mov-call (cosine ~0.37 vs ~0.22)**: the space separates
  different opcode families at all, so the pre-trained weights loaded
  correctly. If both numbers were ~1.0 you'd know the model is frozen /
  identity (likely a loading failure masked by no exception); if they were
  negative or wildly inverted, it'd point at a layer-indexing or
  normalization regression in an upstream change.

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
