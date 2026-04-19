"""Demo - basic-block encoding sanity check.

This is the workflow that was previously the body of run_palmtree.py:

1. Encode a small basic block of x86 instructions.
2. Print the output shape and per-instruction embedding norms (sanity check
   that nothing was clamped to zero).
3. Report a couple of cosine similarities to show the space is non-trivial.

Runs in < 1 s on CPU and is the fastest way to verify the pipeline is
working end-to-end after changing any dependency versions.
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
    emb = c.encode(model, vocab, INSTRUCTIONS)
    print(f"Output shape: {emb.shape}  (N_instructions x embedding_dim)")

    print("\nPer-instruction embedding norms (sanity check):")
    for ins, vec in zip(INSTRUCTIONS, emb):
        print(f"  {ins:35s}  ||emb|| = {np.linalg.norm(vec):.4f}")

    cs_movs = c.cosine_sim(emb[0], emb[1])
    cs_mov_call = c.cosine_sim(emb[0], emb[3])
    print("\nCosine similarities:")
    print(f"  mov-mov     ('mov rbp rdi', 'mov ebx 0x1') = {cs_movs:+.4f}")
    print(f"  mov-call    ('mov rbp rdi', 'call memcpy') = {cs_mov_call:+.4f}")
    print("  (expect mov-mov > mov-call if the space separates opcode families)")
