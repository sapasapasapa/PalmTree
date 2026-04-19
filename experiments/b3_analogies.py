"""B3 - Semantic Analogies in Instruction Space.

Question
--------
word2vec famously supports vector arithmetic (king - man + woman ~ queen).
Does PalmTree's embedding space support analogous operations on
instructions? If embeddings are a linear composition of token meanings, we
should be able to "substitute" a register or opcode by vector subtraction +
addition and land near the expected answer.

Procedure
---------
Form analogy triplets (a, b, c) with an expected target d. Compute
    predicted = embed(a) - embed(b) + embed(c)
and find the rank of d among a large set of candidate instructions. Lower
rank = stronger analogy quality.

Analogies tested
----------------
A) Opcode substitution (register arg held constant):
     "add rax rbx" - "add rax rcx" + "mov rax rcx"  ~  "mov rax rbx"

B) Register substitution (opcode held constant):
     "mov rax rbx" - "rax" + "rdi"  ~  "mov rdi rbx"
   (note: `rax` / `rdi` are single-token inputs - a degenerate case PalmTree
   may not meaningfully represent)

C) Register destination swap:
     "mov rax rbx" - "mov rax rcx" + "mov rdi rcx"  ~  "mov rdi rbx"

What the result reveals
-----------------------
- Strong analogies (d in top-1/top-3) => PalmTree's embedding space has
  learned compositional semantics: opcode + operands = instruction meaning.
- Weak analogies (d in top-20+) => embeddings are holistic gestalts, and
  aggregating them into function-level representations requires a more
  expressive aggregator (attention pooling, learned weights) rather than a
  naive mean.
"""

from __future__ import annotations

import numpy as np

from . import _common as c


# Candidate pool: a varied set of instructions that the analogy target must
# rank-compete against. The more diverse, the stronger the test.
CANDIDATES = [
    "mov rax rbx", "mov rax rcx", "mov rax rdx", "mov rdi rbx", "mov rdi rcx",
    "mov rsi rbx", "mov rsi rdx", "mov rdx rbx", "mov rcx rbx",
    "mov rax [ rbp - 0x8 ]", "mov [ rbp - 0x8 ] rax",
    "add rax rbx", "add rax rcx", "add rdi rbx",
    "sub rax rbx", "sub rax rcx", "imul rax rbx",
    "xor rax rax", "xor rbx rbx",
    "cmp rax rbx", "cmp rax rcx", "test rax rax",
    "push rbx", "push rax", "pop rbx", "pop rax",
    "jmp address", "je address", "jne address",
    "call symbol", "call memcpy", "call malloc", "ret",
    "lea rax [ rbx + rcx * 0x4 ]", "nop",
]


ANALOGIES = [
    {
        "description": "opcode substitution (add -> mov), keeping register args",
        "a": "add rax rbx",
        "b": "add rax rcx",
        "c": "mov rax rcx",
        "expected": "mov rax rbx",
    },
    {
        "description": "destination register swap (rax -> rdi), mov domain",
        "a": "mov rax rbx",
        "b": "mov rax rcx",
        "c": "mov rdi rcx",
        "expected": "mov rdi rbx",
    },
    {
        "description": "source register swap (rbx -> rcx), add domain",
        "a": "add rax rbx",
        "b": "mov rax rbx",
        "c": "mov rax rcx",
        "expected": "add rax rcx",
    },
]


def run(model, vocab):
    c.print_header("B3 - Semantic analogies in instruction space")

    cand_emb = c.encode(model, vocab, CANDIDATES)

    for idx, triplet in enumerate(ANALOGIES, start=1):
        print(f"\n[{idx}] {triplet['description']}")
        print(f"      embed({triplet['a']!r}) - embed({triplet['b']!r}) + embed({triplet['c']!r})")
        print(f"      expected ~ {triplet['expected']!r}")

        a, b, c_vec = c.encode(model, vocab, [triplet["a"], triplet["b"], triplet["c"]])
        predicted = a - b + c_vec

        # Rank all candidates by similarity to predicted
        top = c.top_k_nearest(predicted, cand_emb, CANDIDATES, k=len(CANDIDATES),
                              exclude_self=False)
        rank = next(i for i, (ins, _) in enumerate(top) if ins == triplet["expected"])

        print(f"      rank of expected: {rank + 1}/{len(CANDIDATES)}   "
              f"(top-1 hit? {'YES' if rank == 0 else 'no'})")
        print(f"      top-5 nearest instructions to predicted:")
        for ins, sim in top[:5]:
            marker = " <- expected" if ins == triplet["expected"] else ""
            print(f"         sim={sim:+.3f}  {ins}{marker}")
