"""PalmTree experiment suite.

Each submodule documents and implements one experiment from
`thesis-vault/02 - Atoms/palmtree-experiments.md`. Every module exposes a
single `run(model, vocab)` function that prints its findings. The registry
below maps CLI names to (description, callable).
"""

from . import demo
from . import a1_semantic_clustering
from . import a2_static_inference
from . import a3_layer_wise
from . import a4_vocab_coverage
from . import b2_tokenization
from . import b3_analogies
from . import c2_sequence_length
from . import c4_normalization

EXPERIMENTS = {
    "demo": ("Basic-block encoding demo (replaces the original run_palmtree.py body)", demo.run),
    "a1":   ("A1 - Semantic clustering of instruction families", a1_semantic_clustering.run),
    "a2":   ("A2 - Static inference, contextual weights (invariants)", a2_static_inference.run),
    "a3":   ("A3 - Layer-wise representation quality (paper vs shipped pipeline)", a3_layer_wise.run),
    "a4":   ("A4 - Vocabulary coverage across instruction tiers", a4_vocab_coverage.run),
    "b2":   ("B2 - Fine-grained tokenization vs coarse approaches", b2_tokenization.run),
    "b3":   ("B3 - Semantic analogies in instruction space", b3_analogies.run),
    "c2":   ("C2 - Sequence-length truncation (SEQ_LEN=20 cap)", c2_sequence_length.run),
    "c4":   ("C4 - Normalization sensitivity", c4_normalization.run),
}
