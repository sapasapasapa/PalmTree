"""A1 - Semantic Clustering of Instruction Families.

Question
--------
Does PalmTree's embedding space organize instructions by semantic category
(what they do), or just by opcode (the leading mnemonic)?

Procedure
---------
Encode a set of x86 instructions spanning four distinct semantic families:
data movement, arithmetic, comparison, and control flow. Then:

1. Compute the intra-family mean cosine similarity (tightness of each cluster).
2. Compute the inter-family mean cosine similarity (how close clusters are
   to each other).
3. Compute cluster purity: for every instruction, check whether its single
   nearest neighbor in the full set belongs to the same family.
4. Classify "probe" instructions (`lea`, a stack-local store) against the
   families by nearest-neighbor vote.

No t-SNE/UMAP is drawn (sklearn + matplotlib aren't installed in the pinned
venv) - instead we report the numbers that those plots would visually convey.

What the result reveals
-----------------------
- If intra >> inter and purity is high, PalmTree has learned a genuinely
  semantic space where instructions group by function.
- If intra ~ inter or purity is near random (1/n_families), the space is
  mostly driven by surface tokens and not semantic category.
- If `lea rax [rbx + rcx*4]` classifies as arithmetic rather than data
  movement, the model has inferred that `lea` is effectively an arithmetic
  instruction - a non-trivial semantic deduction.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import _common as c


FAMILIES = {
    "data_movement": [
        "mov rax rbx",
        "mov rdi rsp",
        "mov rax [ rbp - 0x8 ]",
        "mov [ rax ] rcx",
        "push rbx",
        "pop rcx",
    ],
    "arithmetic": [
        "add rax rbx",
        "sub rax rcx",
        "imul rdx rbx",
        "shr rax 0x4",
        "xor rax rax",
        "inc rdx",
    ],
    "comparison": [
        "cmp rax rbx",
        "test rax rax",
        "cmp eax 0x0",
        "test rcx rcx",
    ],
    "control_flow": [
        "jmp address",
        "je address",
        "jne address",
        "call symbol",
        "ret",
    ],
}

PROBES = [
    ("lea rax [ rbx + rcx * 0x4 ]", "lea (ALU-like 'mov')"),
    ("mov [ rbp - 0x8 ] rax",       "store to local"),
    ("xchg rax rbx",                "register swap"),
]


def run(model, vocab):
    c.print_header("A1 - Semantic clustering of instruction families")

    all_instructions, all_labels = [], []
    for family, items in FAMILIES.items():
        all_instructions.extend(items)
        all_labels.extend([family] * len(items))

    emb = c.encode(model, vocab, all_instructions)
    sim = c.pairwise_cosine(emb)

    # --- intra vs inter family means -------------------------------------
    c.print_subheader("Intra vs inter-family mean cosine similarity")
    family_names = list(FAMILIES.keys())
    for f in family_names:
        idxs = [i for i, l in enumerate(all_labels) if l == f]
        intra = [sim[i, j] for i in idxs for j in idxs if i < j]
        others = [i for i, l in enumerate(all_labels) if l != f]
        inter = [sim[i, j] for i in idxs for j in others]
        print(f"  {f:16s}  intra={np.mean(intra):+.3f}  inter={np.mean(inter):+.3f}  "
              f"gap={np.mean(intra) - np.mean(inter):+.3f}")

    # --- cluster purity (1-NN) -------------------------------------------
    c.print_subheader("1-NN cluster purity (is the nearest neighbor in the same family?)")
    correct = 0
    for i, ins in enumerate(all_instructions):
        row = sim[i].copy()
        row[i] = -np.inf
        nn = int(np.argmax(row))
        hit = all_labels[nn] == all_labels[i]
        correct += hit
        flag = "OK" if hit else "!!"
        print(f"  [{flag}] {ins:35s} -> {all_instructions[nn]:35s} "
              f"({all_labels[nn]}, sim={row[nn]:+.3f})")
    total = len(all_instructions)
    print(f"\n  Purity: {correct}/{total} = {correct / total:.2%}  "
          f"(random baseline = 1/{len(family_names)} = {1 / len(family_names):.2%})")

    # --- probe classification --------------------------------------------
    c.print_subheader("Probe classification (top-3 nearest across all families)")
    probe_emb = c.encode(model, vocab, [p[0] for p in PROBES])
    for (probe, description), p_vec in zip(PROBES, probe_emb):
        print(f"\n  probe: {probe}  ({description})")
        neighbors = c.top_k_nearest(p_vec, emb, list(zip(all_instructions, all_labels)), k=3)
        votes = Counter()
        for (ins, fam), s in neighbors:
            print(f"      sim={s:+.3f}  [{fam:14s}]  {ins}")
            votes[fam] += 1
        predicted_family = votes.most_common(1)[0][0]
        print(f"    => nearest-family vote: {predicted_family}")
