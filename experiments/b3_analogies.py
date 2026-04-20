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
----------------A
) Opcode substitution (register arg held constant):
     "add rax rbx" - "add rax rcx" + "mov rax rcx"  ~  "mov rax rbx"

B) Register substitution (opcode held constant):
     "mov rax rbx" - "rax" + "rdi"  ~  "mov rdi rbx"
   (note: `rax` / `rdi` are single-token inputs - a degenerate case PalmTree
   may not meaningfully represent)

C) Register destination swap:
     "mov rax rbx" - "mov rax rcx" + "mov rdi rcx"  ~  "mov rdi rbx"

What the experiment reveals
---------------------------
(Purely hypothetical branches. Each bullet is "IF you see X -> it means Y",
covering possibilities that may or may not materialize in any given run.
The actual numbers from this run are in "Observed results" below.)

- **IF the expected target hits rank 1 (top-1) across most analogies**:
  PalmTree's embedding space is approximately linear in opcode and
  operand components. You can do token-level algebra on embeddings to
  compose queries. Example utility: "show me all registers substituted
  with rdi" becomes a vector-arithmetic operation, not a
  string-manipulation-then-re-encode round trip.
- **IF the expected target lands at rank 2, beaten by the `c` term**:
  classic word2vec analogy pitfall - the predicted vector sits very
  close to `c` (its largest constituent), so `c` wins the cosine race
  even when the analogy is mostly right. Mitigation: exclude the input
  terms {a, b, c} from the candidate pool before ranking, or use
  3CosMul instead of 3CosAdd for ranking. Rank-2 is still evidence of
  meaningful linear structure.
- **IF the expected target lands in top-20 or worse**: embeddings are
  holistic, not compositional. Consequence for downstream design:
  aggregating instructions into function embeddings via NAIVE MEAN
  (sum then divide) will lose signal. You need a learned aggregator
  (attention pooling, transformer on top of PalmTree embeddings) to
  recover function-level similarity. Exactly the gap that later models
  like jTrans address by end-to-end fine-tuning on function pairs.
- **IF top-5 is dominated by instructions sharing the same register or
  opcode as `c`**: the analogy is "moving along a register axis". When
  that axis is long (c term dominates the vector), the mechanism
  degenerates into "find things similar to c". When the axis is short
  (a-b is comparable in magnitude to c), analogies are cleaner. Tuning
  analogies often requires similar-magnitude operands.
- **IF rank varies wildly across the three analogies (e.g. one top-1,
  one rank 30)**: linearity is partial and directional - some axes
  (opcode substitution, destination register) are well-learned, others
  (source register, address mode) are not. A thesis should report the
  full rank distribution, not a single aggregate number.

Observed results
----------------
- **[1] opcode substitution (add -> mov): rank 2/35**. The expected target
  `mov rax rbx` scored 0.749, beaten by the `c` input term `mov rax rcx`
  at 0.860. Classic word2vec "c dominates" pitfall: subtract `add rax rcx`
  from `add rax rbx` leaves a vector close to `mov rax rcx` rather than
  neatly translated to `mov rax rbx`. Near-miss but the analogy structure
  is present (all five top-5 candidates share either the mov opcode or
  the rbx register with the expected answer).
- **[2] dest register swap (rax -> rdi): rank 2/35**. Expected `mov rdi rbx`
  at 0.678, beaten by `c` = `mov rdi rcx` at 0.697. Same failure mode as
  [1]: the c-term dominance. But crucially the top-4 candidates all share
  the target's rdi destination, so the "swap destination to rdi" axis
  was correctly extracted.
- **[3] source register swap (rbx -> rcx): rank 1/35 - TOP-1 HIT**. Expected
  `add rax rcx` wins at 0.750 (very narrowly over `add rax rbx` at 0.746).
  When the analogy axis (the a-b vector = `add - mov` opcode delta) is
  substantial and the c term (`mov rax rcx`) doesn't overlap with the
  expected target in opcode, the analogy lands cleanly.
- **Aggregate: 1/3 top-1, 3/3 top-2. Rank distribution = [2, 2, 1]**. This
  is PARTIAL linearity: enough structure to support interpretability
  tooling (e.g. "find all variants substituting rdi" via vector arithmetic)
  if you exclude input terms from the candidate pool, but not enough that
  raw 3CosAdd is a reliable retrieval method. For function-level tasks
  you will still want a learned aggregator.

Bottom line for thesis: report the rank distribution across several
analogies, not a single top-1 hit rate. PalmTree shows PARTIAL linearity
- enough to enable some interpretability tools, but not enough to skip
fine-tuning for a function-similarity task.
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

    # Encode the candidate pool ONCE outside the loop; we reuse the same
    # (N, 128) array for every analogy. cand_emb[i] aligns with CANDIDATES[i].
    cand_emb = c.encode(model, vocab, CANDIDATES)

    for idx, triplet in enumerate(ANALOGIES, start=1):
        print(f"\n[{idx}] {triplet['description']}")
        print(f"      embed({triplet['a']!r}) - embed({triplet['b']!r}) + embed({triplet['c']!r})")
        print(f"      expected ~ {triplet['expected']!r}")

        # Encode the three analogy inputs together (single forward pass, not
        # three). Python multi-assignment unpacks the (3, 128) array into
        # three individual (128,) row vectors.
        a, b, c_vec = c.encode(model, vocab, [triplet["a"], triplet["b"], triplet["c"]])
        # Classic word2vec analogy vector: a - b + c. Reads as "b is to a
        # as c is to ?". We isolate the a-b delta (the conceptual offset)
        # and apply it to c to guess the missing fourth term.
        predicted = a - b + c_vec

        # Rank every candidate by cosine similarity to `predicted`.
        # exclude_self=False because `predicted` is a new synthesized
        # vector, not a row of cand_emb. k=len(CANDIDATES) gives the full
        # ranking so we can find where the expected target lands.
        top = c.top_k_nearest(predicted, cand_emb, CANDIDATES, k=len(CANDIDATES),
                              exclude_self=False)
        # Walk the ranked list to find the expected target's position.
        # next() with a generator returns the first index that matches;
        # since we pass every candidate, the target is guaranteed present.
        rank = next(i for i, (ins, _) in enumerate(top) if ins == triplet["expected"])

        print(f"      rank of expected: {rank + 1}/{len(CANDIDATES)}   "
              f"(top-1 hit? {'YES' if rank == 0 else 'no'})")
        print(f"      top-5 nearest instructions to predicted:")
        for ins, sim in top[:5]:
            marker = " <- expected" if ins == triplet["expected"] else ""
            print(f"         sim={sim:+.3f}  {ins}{marker}")
