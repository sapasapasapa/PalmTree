"""Entry point for PalmTree on Apple Silicon / modern PyTorch.

This script has two responsibilities:

1. **Pickle compatibility shim.** The released pre-trained model was pickled
   against a package named `bert_pytorch`, but the source in this repo lives
   under `palmtree`. Before any `torch.load` runs, we register alias entries
   in `sys.modules` so the pickle deserializes against the renamed package.

2. **Experiment dispatcher.** A CLI that loads the model once and runs either
   the original demo or one of the experiments in `experiments/` (a set of
   hands-on probes that characterize what PalmTree's embeddings capture and
   where they break down — see thesis-vault note `palmtree-experiments.md`).

Usage:
    source .venv/bin/activate
    python run_palmtree.py                 # list available experiments
    python run_palmtree.py demo            # basic-block encoding demo
    python run_palmtree.py a1              # A1 - semantic clustering
    python run_palmtree.py all             # run every experiment in order
    python run_palmtree.py --help
"""

import sys
import os

# --- Pickle module shim (MUST run before any torch.load) -------------------
# The pickle inside pre-trained_model/palmtree/transformer.ep19 references
# classes under `bert_pytorch.*`. Redirect those names to the `palmtree.*`
# package that actually ships in src/.

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pre-trained_model"))

import palmtree as _palmtree_pkg  # noqa: E402
sys.modules["bert_pytorch"] = _palmtree_pkg

# Eagerly import every submodule the pickle may reference.
import palmtree.dataset.vocab            # noqa: E402,F401
import palmtree.model.bert               # noqa: E402,F401
import palmtree.model.transformer        # noqa: E402,F401
import palmtree.model.attention.single   # noqa: E402,F401
import palmtree.model.attention.multi_head  # noqa: E402,F401
import palmtree.model.embedding.bert     # noqa: E402,F401
import palmtree.model.embedding.position # noqa: E402,F401
import palmtree.model.embedding.segment  # noqa: E402,F401
import palmtree.model.embedding.token    # noqa: E402,F401
import palmtree.model.utils.feed_forward # noqa: E402,F401
import palmtree.model.utils.gelu         # noqa: E402,F401
import palmtree.model.utils.layer_norm   # noqa: E402,F401
import palmtree.model.utils.sublayer     # noqa: E402,F401
import palmtree                          # noqa: E402

_ALIASES = {
    "bert_pytorch.dataset": palmtree.dataset,
    "bert_pytorch.dataset.vocab": palmtree.dataset.vocab,
    "bert_pytorch.model": palmtree.model,
    "bert_pytorch.model.bert": palmtree.model.bert,
    "bert_pytorch.model.transformer": palmtree.model.transformer,
    "bert_pytorch.model.attention": palmtree.model.attention,
    "bert_pytorch.model.attention.single": palmtree.model.attention.single,
    "bert_pytorch.model.attention.multi_head": palmtree.model.attention.multi_head,
    "bert_pytorch.model.embedding": palmtree.model.embedding,
    "bert_pytorch.model.embedding.bert": palmtree.model.embedding.bert,
    "bert_pytorch.model.embedding.position": palmtree.model.embedding.position,
    "bert_pytorch.model.embedding.segment": palmtree.model.embedding.segment,
    "bert_pytorch.model.embedding.token": palmtree.model.embedding.token,
    "bert_pytorch.model.utils": palmtree.model.utils,
    "bert_pytorch.model.utils.feed_forward": palmtree.model.utils.feed_forward,
    "bert_pytorch.model.utils.gelu": palmtree.model.utils.gelu,
    "bert_pytorch.model.utils.layer_norm": palmtree.model.utils.layer_norm,
    "bert_pytorch.model.utils.sublayer": palmtree.model.utils.sublayer,
}
sys.modules.update(_ALIASES)

# --- Normal imports (safe now that the shim is installed) ------------------
import argparse  # noqa: E402
import pickle    # noqa: E402
import torch     # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(__file__), "pre-trained_model", "palmtree")
MODEL_PATH = os.path.join(MODEL_DIR, "transformer.ep19")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab")


def load_model():
    """Load the pre-trained PalmTree model and its WordVocab onto CPU.

    Returns (model, vocab). The model is put into eval mode so dropout and
    other training-only behaviors are disabled.
    """
    print(f"Loading vocab from {VOCAB_PATH}")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"  Vocab size: {len(vocab)}")

    print(f"Loading model from {MODEL_PATH}")
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()

    n_layers = len(model.transformer_blocks)
    hidden = model.hidden
    print(f"  Model loaded: BERT n_layers={n_layers}, hidden={hidden} (CPU mode)")
    return model, vocab


def main():
    parser = argparse.ArgumentParser(
        description="Run PalmTree experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        default=None,
        help="Experiment name. Omit to list available experiments.",
    )
    args = parser.parse_args()

    # Lazy import so that `--help` does not trigger model loading.
    from experiments import EXPERIMENTS

    if args.experiment is None:
        print("Available experiments:\n")
        for name, (desc, _) in EXPERIMENTS.items():
            print(f"  {name:6s}  {desc}")
        print("\n  all     Run every experiment in order")
        print(f"\nRun with:  python {os.path.basename(__file__)} <name>")
        return

    if args.experiment not in EXPERIMENTS and args.experiment != "all":
        parser.error(
            f"unknown experiment '{args.experiment}'. "
            f"Known: {', '.join(EXPERIMENTS)} or 'all'."
        )

    model, vocab = load_model()

    if args.experiment == "all":
        for name, (desc, fn) in EXPERIMENTS.items():
            print(f"\n\n{'#' * 70}\n# {name}: {desc}\n{'#' * 70}")
            fn(model, vocab)
        return

    _, fn = EXPERIMENTS[args.experiment]
    fn(model, vocab)


if __name__ == "__main__":
    main()
