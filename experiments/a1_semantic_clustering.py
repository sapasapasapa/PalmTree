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

What the experiment reveals
---------------------------
(Purely hypothetical branches. Each bullet is "IF you see X -> it means Y",
covering possibilities that may or may not materialize in any given run.
The actual numbers from this run are in "Observed results" below.)

- **IF all four families show large positive gaps (intra >> inter) AND
  purity > 80%**: PalmTree has learned a genuinely semantic space.
  Downstream similarity tasks (function cloning, malware family
  classification) will have strong structural signal to exploit - a linear
  classifier on top of the embeddings should work out of the box.
- **IF overall gap is near zero OR purity is near the 25% random baseline**:
  the space is driven mainly by surface token co-occurrence, not semantic
  category. Practical consequence: any downstream task that assumes
  "instructions doing similar things have similar embeddings" will need
  fine-tuning on labeled pairs. Frozen-model transfer will underperform.
- **IF data_movement specifically shows a low or negative intra-gap while
  other families show positive gaps**: `mov` variants are far apart in the
  space because operand patterns (reg-reg vs memory-load vs memory-store)
  dominate the embedding. This is a *feature* for operand-aware downstream
  tasks (stack-frame analysis, call-convention inference) but a *bug* for
  coarse opcode-family classification. If your downstream task treats all
  movs as interchangeable, pre-cluster or aggregate them yourself.
- **IF instead data_movement shows a high positive gap like the others**:
  the model treats the `mov` mnemonic as a strong family signal and
  operand details are subordinate - which means you would LOSE the
  load-vs-store, arg-setup-vs-local-var distinctions that matter for
  stack-frame analysis. Useful for mnemonic-family classification, but
  worse for role-aware tasks.
- **IF `lea` is classified as arithmetic**: the model has inferred a
  non-trivial semantic property - that `lea` is an ALU op, not data
  movement. Exactly the deduction you want for reverse engineering: the
  model learned from behavior patterns in Coreutils that `lea` appears
  near adds/imuls even though its mnemonic says "load".
- **IF instead `lea` lands in data_movement by nearest-neighbor vote**:
  the model is opcode-biased (mnemonic "lea" -> "looks like a mov"). You
  would lose a useful deduction for free; add explicit opcode-usage
  features if downstream tasks depend on ALU-op recognition.
- **IF purity is 60-70%**: typical "good but not great". Downstream
  classifiers can reach higher accuracy by pooling multiple instructions
  (reducing per-sample noise) and/or adding explicit opcode features on
  top of the embedding.
- **IF purity < 30% (near random)**: something is broken in the pipeline
  or the model file. Re-check the pickle shim and the vocab load.

Observed results
----------------
- **Intra/inter gaps per family**: arithmetic +0.176, comparison +0.183,
  control_flow +0.204, data_movement -0.009. Three families show the
  predicted "tighter inside than across" pattern; data_movement does NOT.
  The negative gap confirms mov variants are operand-dominated, not
  opcode-dominated - use this as the canonical example when arguing that
  PalmTree is an operand-aware, not a mnemonic-only, embedding.
- **1-NN purity = 14/21 = 66.7% vs 25% random baseline (2.67x lift)**:
  solid-but-imperfect semantic structure. The 7 misses are diagnostic:
  `mov rax rbx` -> `add rax rbx` (register copy ~ ALU on same regs),
  `mov rdi rsp` -> `imul rdx rbx` (function-arg setup classified as
  arithmetic), `pop rcx` -> `ret` (stack-pop classified as control flow -
  arguably semantically correct), `add rax rbx` -> `mov rax rbx`
  (symmetric of the first miss), `cmp rax rbx` and `cmp eax 0x0` -> mov/add
  (comparison bleeding into arithmetic), and `call symbol` -> `inc rdx`
  (call misfiled). Actionable: the confusion is mostly arithmetic <-> data
  movement <-> comparison at the register-reg boundary; control_flow is
  nearly clean (jmp/je/jne/ret all correct).
- **Probes**: `lea` -> arithmetic (all three nearest neighbors are
  arithmetic: add/xor/shr). The model HAS learned lea is an ALU op.
  `mov [rbp-0x8] rax` -> data_movement (top-3 all data_movement).
  `xchg rax rbx` -> data_movement with sim 0.865 to `mov rax rbx` -
  excellent: the model knows xchg is a mov-family operation despite being
  a rare opcode.
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

    # Flatten the dict-of-lists into parallel arrays: instructions[i] is
    # labelled by labels[i]. Keeping these aligned lets us reference family
    # membership by row index into the embedding matrix below.
    all_instructions, all_labels = [], []
    for family, items in FAMILIES.items():
        all_instructions.extend(items)
        all_labels.extend([family] * len(items))

    # emb is (N, 128): one embedding vector per instruction.
    emb = c.encode(model, vocab, all_instructions)
    # sim is (N, N): cosine similarity between every pair of instructions.
    # sim[i, j] == sim[j, i], and sim[i, i] == 1.0 (self-similarity).
    sim = c.pairwise_cosine(emb)

    # --- intra vs inter family means -------------------------------------
    c.print_subheader("Intra vs inter-family mean cosine similarity")
    family_names = list(FAMILIES.keys())
    for f in family_names:
        # Row indices of the instructions belonging to this family.
        idxs = [i for i, l in enumerate(all_labels) if l == f]
        # intra: similarities between DIFFERENT members of the same family.
        # `i < j` filter avoids counting (i, j) and (j, i) twice, and skips
        # the self-pair (i, i) which would always be 1.0 and bias the mean.
        intra = [sim[i, j] for i in idxs for j in idxs if i < j]
        others = [i for i, l in enumerate(all_labels) if l != f]
        # inter: similarities between each member of this family and every
        # member of every OTHER family. No double-count filter needed because
        # we iterate one direction only (family -> others).
        inter = [sim[i, j] for i in idxs for j in others]
        # "gap" is the key diagnostic. Positive gap means the family is more
        # internally coherent than it is similar to the rest; negative gap
        # means the family is LESS tight than random cross-family pairs
        # (observed here for data_movement).
        print(f"  {f:16s}  intra={np.mean(intra):+.3f}  inter={np.mean(inter):+.3f}  "
              f"gap={np.mean(intra) - np.mean(inter):+.3f}")

    # --- cluster purity (1-NN) -------------------------------------------
    c.print_subheader("1-NN cluster purity (is the nearest neighbor in the same family?)")
    correct = 0
    for i, ins in enumerate(all_instructions):
        # Work on a copy because we're about to mutate the row.
        row = sim[i].copy()
        # Set self-similarity to -inf so argmax cannot pick index i. This is
        # the standard trick for "nearest neighbor excluding self".
        row[i] = -np.inf
        # np.argmax returns the index of the maximum value. int() casts the
        # numpy scalar back to a Python int for cleaner printing + indexing.
        nn = int(np.argmax(row))
        hit = all_labels[nn] == all_labels[i]
        correct += hit
        flag = "OK" if hit else "!!"
        print(f"  [{flag}] {ins:35s} -> {all_instructions[nn]:35s} "
              f"({all_labels[nn]}, sim={row[nn]:+.3f})")
    total = len(all_instructions)
    # Compare against the random baseline (1/num_families) so the reader can
    # gauge "is 67% actually good?" - 67% against a 25% baseline is a ~2.7x
    # lift over random, not a hard-pass on its own.
    print(f"\n  Purity: {correct}/{total} = {correct / total:.2%}  "
          f"(random baseline = 1/{len(family_names)} = {1 / len(family_names):.2%})")

    # --- probe classification --------------------------------------------
    c.print_subheader("Probe classification (top-3 nearest across all families)")
    # Encode each probe once. probe_emb is (n_probes, 128).
    probe_emb = c.encode(model, vocab, [p[0] for p in PROBES])
    for (probe, description), p_vec in zip(PROBES, probe_emb):
        print(f"\n  probe: {probe}  ({description})")
        # top_k_nearest ranks all instructions by cosine sim to the probe.
        # We pack (instruction, family) tuples as labels so we can recover
        # family membership for each neighbor.
        neighbors = c.top_k_nearest(p_vec, emb, list(zip(all_instructions, all_labels)), k=3)
        # Counter tallies family votes across the top-3 neighbors. Ties are
        # broken by Counter.most_common returning items in insertion order.
        votes = Counter()
        for (ins, fam), s in neighbors:
            print(f"      sim={s:+.3f}  [{fam:14s}]  {ins}")
            votes[fam] += 1
        predicted_family = votes.most_common(1)[0][0]
        print(f"    => nearest-family vote: {predicted_family}")
