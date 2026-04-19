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

What the result reveals
-----------------------
- If layer 11 > layer 12 on discrimination, the paper's advice holds and the
  shipped `UsableTransformer.encode` is objectively worse than using
  `model.encode()` (or this module's `encode(..., layer=11)` helper).
- Early layers (0-4) usually show weak discrimination but can occasionally
  beat deep layers for specific tasks - informative when considering
  distillation or a shallower deployment variant.
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
    sim = c.pairwise_cosine(mat)
    n = sim.shape[0]
    if n < 2:
        return float("nan")
    return float(sim[np.triu_indices(n, 1)].mean())


def _mean_cross(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_u = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_u = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
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
    for layer in range(n_layers + 1):
        sim_emb = c.encode(model, vocab, SIMILAR_GROUP, layer=layer)
        dis_emb = c.encode(model, vocab, DISSIMILAR_GROUP, layer=layer)
        intra = _mean_pairwise(sim_emb)
        inter = _mean_cross(sim_emb, dis_emb)
        gap = intra - inter
        results.append((layer, intra, inter, gap))
        annotation = ""
        if layer == n_layers:
            annotation = "  <- UsableTransformer.forward()"
        elif layer == n_layers - 1:
            annotation = "  <- paper's recommendation / bert.py .encode()"
        print(f"  {layer:>6d}  {intra:+.4f}  {inter:+.4f}  {gap:+.4f}{annotation}")

    # Summary: which layer wins?
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
