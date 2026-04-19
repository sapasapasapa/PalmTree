"""A3 - Layer-wise Representation Quality.

Question
--------
The PalmTree paper recommends mean-pooling the **second-to-last** BERT layer
(layer 11 of 12), arguing the final layer over-specializes to the
pre-training objectives. How much does layer choice actually matter, and
what does the **shipped** pipeline actually use?

Paper-vs-code discrepancy (important)
-------------------------------------
`src/palmtree/model/bert.py` defines two forward variants:

    def forward(self, x, segment_info):      # runs ALL 12 blocks
        ...
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)
        return x

    def encode(self, x, segment_info):        # stops before last block
        ...
        for transformer in self.transformer_blocks[:-1]:
            x = transformer.forward(x, mask)
        return x

But the `UsableTransformer` wrapper in `eval_utils.py` calls `.forward()`,
not `.encode()`. So the shipped library actually returns the FINAL layer's
representation - contradicting the paper. This experiment exposes the
discrepancy quantitatively.

Procedure
---------
Extract mean-pooled embeddings from each layer index 0..N (0 = after token/
position/segment embedding, before any transformer block). For each layer:

1. Compute mean cosine similarity within a semantically-similar group
   (three arithmetic instructions).
2. Compute mean cosine similarity across semantically-dissimilar pairs
   (arithmetic vs control flow).
3. Report the discrimination ratio = mean_intra - mean_inter. Higher is
   better: that layer separates "same" from "different" more strongly.

What the result reveals - concrete consequences
-------------------------------------------------
- **Layer 11 beats layer 12 on the gap metric**: the paper's advice holds.
  Actionable fix in the shipped code: change `eval_utils.py:92` from
  `self.model.forward(...)` to `self.model.encode(...)`. Existing consumers
  get a free quality bump without retraining. This would be a one-line PR
  upstream that affects every downstream user.
- **Layer 12 beats layer 11 (observed on this specific test)**: the paper's
  claim does NOT generalize to raw cosine-discrimination between
  semantic groups. There are two possible explanations: (a) the paper's
  "over-specialization" argument is about transfer-learning AUC on
  downstream tasks (which is NOT what we measure here), or (b) the
  specific instruction set used in the test biases which layer wins. Either
  way: do NOT blindly trust "second-to-last layer is better" as a universal
  claim. For your own downstream task, measure layer choice on a held-out
  set of task-labeled pairs. This experiment gives you the infrastructure
  (layer-stopping encode) to do exactly that.
- **Early layers (0-4) have low discrimination**: expected, since the first
  couple of blocks mostly propagate the raw embeddings. If, however, an
  early layer matches layer 11 on a specific task, that opens the door to
  a SHALLOWER deployed model. You could fine-tune only the first k blocks
  and discard the rest - useful for embedding millions of instructions on
  constrained hardware. This is the distillation / pruning signal.
- **Intra and inter both drop monotonically as layer increases**: the
  representation becomes MORE spread out as depth grows (each block
  pushes tokens apart in the 128-dim space). This is characteristic of
  BERT-style MLM training. One practical implication: any cosine threshold
  you tune ("accept pairs with cosine > T") MUST be tuned per layer -
  thresholds do not transfer across depths.
- **Negative gap at some layer**: that layer actively CONFUSES arithmetic
  and control-flow in its representation. Almost always a sign of a
  catastrophic training issue, a bad load, or a metric bug. Worth re-
  checking the pickle shim and the model file's integrity.
"""

from __future__ import annotations

import numpy as np

from . import _common as c


SIMILAR_GROUP = [   # arithmetic / ALU
    "add rax rbx",
    "sub rax rcx",
    "imul rdx rbx",
    "shr rax 0x4",
    "xor rax rax",
]

DISSIMILAR_GROUP = [  # control flow - semantically far from arithmetic
    "jmp address",
    "je address",
    "call symbol",
    "ret",
]


def _mean_pairwise(mat):
    """Mean of the unique pairwise cosine similarities within `mat`.

    "Unique" means i<j only: skip the diagonal (self-sim is always 1.0 and
    would inflate the mean) and skip (j, i) duplicates of (i, j).
    """
    sim = c.pairwise_cosine(mat)
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    # np.triu_indices(n, k=1) gives (rows, cols) for the strict upper
    # triangle of an n-by-n matrix. sim[those indices] is a 1-D array of
    # length n*(n-1)/2 containing each unique pair once.
    return float(sim[np.triu_indices(n, 1)].mean())


def _mean_cross(a, b):
    """Mean cosine similarity between every row of `a` and every row of `b`.

    No within-matrix pairs, no diagonal filtering - this is purely
    cross-group distance, so every pair (i, j) with i in a, j in b counts.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Normalize each row to unit length so dot products equal cosines.
    a_u = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_u = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    # (Na, D) @ (D, Nb) = (Na, Nb) cross-similarity matrix.
    # .mean() averages all Na*Nb entries.
    return float((a_u @ b_u.T).mean())


def run(model, vocab):
    c.print_header("A3 - Layer-wise representation quality")

    n_layers = len(model.transformer_blocks)
    print(f"Model has {n_layers} transformer blocks. Probing layers 0..{n_layers}.\n")
    print(f"  Layer 0  = after token+position+segment embedding, no attention applied")
    print(f"  Layer k  = after the k-th transformer block")
    print(f"  Layer {n_layers} = final layer (what UsableTransformer.forward returns)")
    print(f"  Layer {n_layers - 1} = second-to-last layer (what the paper recommends, and")
    print(f"            what bert.py's .encode() method returns)")

    c.print_subheader("Discrimination = mean_intra(arithmetic) - mean_inter(arithmetic vs control_flow)")
    print(f"  {'layer':>6s}  {'intra':>8s}  {'inter':>8s}  {'gap':>8s}")
    results = []
    # Loop over layer = 0..n_layers INCLUSIVE. layer=0 means embedding only,
    # layer=n_layers means all blocks applied (the shipped forward pass).
    for layer in range(n_layers + 1):
        # c.encode with layer=k: re-walks the forward pass and stops after
        # block k, returning hidden states mean-pooled across seq_len.
        sim_emb = c.encode(model, vocab, SIMILAR_GROUP, layer=layer)
        dis_emb = c.encode(model, vocab, DISSIMILAR_GROUP, layer=layer)
        intra = _mean_pairwise(sim_emb)
        inter = _mean_cross(sim_emb, dis_emb)
        # gap is our "how well does this layer separate semantic groups"
        # single-number metric. Higher is better. Can be negative if the
        # layer confuses the two groups.
        gap = intra - inter
        results.append((layer, intra, inter, gap))
        annotation = ""
        if layer == n_layers:
            annotation = "  <- UsableTransformer.forward()"
        elif layer == n_layers - 1:
            annotation = "  <- paper's recommendation / bert.py .encode()"
        print(f"  {layer:>6d}  {intra:+.4f}  {inter:+.4f}  {gap:+.4f}{annotation}")

    # Summary: which layer wins?
    # max(..., key=lambda r: r[3]) picks the tuple with the largest gap
    # (results[3] is the gap column). r[3] is also destructured below as
    # best_gap. The underscores ignore intra/inter - we only care about
    # which layer index is best.
    best_layer, _, _, best_gap = max(results, key=lambda r: r[3])
    shipped_gap = results[n_layers][3]
    paper_gap = results[n_layers - 1][3]

    c.print_subheader("Summary")
    print(f"  best-discriminating layer:       {best_layer}  (gap = {best_gap:+.4f})")
    print(f"  paper recommendation (layer {n_layers - 1}): gap = {paper_gap:+.4f}")
    print(f"  shipped pipeline     (layer {n_layers}): gap = {shipped_gap:+.4f}")
    delta = paper_gap - shipped_gap
    sign = "+" if delta > 0 else ""
    print(f"  paper-vs-shipped delta: {sign}{delta:.4f}  "
          f"({'paper layer is better' if delta > 0 else 'shipped layer is better'})")
