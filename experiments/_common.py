"""Shared utilities for PalmTree experiments.

Every experiment uses the same tokenize-pad-encode pipeline. Centralizing it
here (a) guarantees consistency across experiments, and (b) lets A3 plug in
its own layer-stopping forward pass without duplicating the input prep.
"""

from __future__ import annotations

import numpy as np
import torch

# PalmTree pre-training fixed SEQ_LEN=20; changing this for inference-only
# is safe (longer context), but the position embedding table is trained for
# length 20 so extending may degrade quality. Experiment C2 probes this.
DEFAULT_SEQ_LEN = 20


def prepare_input(vocab, instructions, seq_len=DEFAULT_SEQ_LEN):
    """Tokenize, wrap in <SOS>/<EOS>, and pad to seq_len.

    Matches the logic in eval_utils.UsableTransformer.encode so experiments
    use the exact same pipeline as the shipped API.

    Returns (sequence_ids, segment_labels) as torch.LongTensors of shape
    (batch, seq_len).
    """
    segment_label = []
    sequence = []
    for ins in instructions:
        tokens = ins.split(" ")
        # Segment labels are BERT's "which sentence is this token in" mask.
        # PalmTree treats each instruction as a single sentence -> all 1s for
        # real content positions, 0s for padding. +2 counts the <SOS>/<EOS>
        # that we prepend/append on the next two lines.
        label = [1] * (len(tokens) + 2)
        # vocab.to_seq maps each whitespace-split token to its integer id via
        # vocab.stoi; unknown tokens get vocab.unk_index (= 1).
        ids = vocab.to_seq(ins)
        # Wrap with special tokens; indices are fixed at pre-training time:
        # 3 = <SOS> (start-of-sequence), 2 = <EOS> (end-of-sequence).
        ids = [3] + ids + [2]

        # Truncate if too long, right-pad with 0 (PAD index) otherwise. Both
        # the id and the segment-label tensors get identical shape treatment.
        if len(label) > seq_len:
            segment_label.append(label[:seq_len])
        else:
            segment_label.append(label + [0] * (seq_len - len(label)))
        if len(ids) > seq_len:
            sequence.append(ids[:seq_len])
        else:
            sequence.append(ids + [0] * (seq_len - len(ids)))

    # torch.LongTensor == int64 tensor. Shape (batch, seq_len) for both.
    return torch.LongTensor(sequence), torch.LongTensor(segment_label)


@torch.no_grad()  # disables autograd -> no gradients tracked; saves memory at eval time
def encode(model, vocab, instructions, layer=None, seq_len=DEFAULT_SEQ_LEN, pool="mean"):
    """Encode a list of instructions to embeddings.

    Parameters
    ----------
    layer : int | None
        None   => use model.forward() (runs all 12 transformer blocks; matches
                  the UsableTransformer wrapper in eval_utils.py).
        0      => return the post-embedding hidden state, before any block.
        1..N   => return the hidden state after block `layer`.
        N=len(model.transformer_blocks) is the same as layer=None.
    pool : "mean" | "none"
        "mean" collapses the seq_len dimension and returns (batch, hidden).
        "none" returns (batch, seq_len, hidden) for deeper analysis.
    """
    # x:            (batch, seq_len) int64 token IDs
    # segment_info: (batch, seq_len) int64 segment labels (1 = content, 0 = pad)
    x, segment_info = prepare_input(vocab, instructions, seq_len)

    if layer is None:
        # Default path: run ALL transformer blocks via the stock forward
        # pass. Output shape: (batch, seq_len, hidden_dim).
        hidden = model.forward(x, segment_info)
    else:
        # Custom path for A3: rebuild the forward pass and stop early.
        # Mask construction is copied verbatim from bert.py:39 so behavior
        # matches the stock pipeline.
        #   (x > 0)                                  -> bool (batch, seq_len),  True on content
        #   .unsqueeze(1)                            -> (batch, 1, seq_len)
        #   .repeat(1, x.size(1), 1)                 -> (batch, seq_len, seq_len)
        #   .unsqueeze(1)                            -> (batch, 1, seq_len, seq_len)
        # The final shape broadcasts over attention heads. A query at any
        # position may attend only to KEY positions where x > 0 (non-pad).
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # model.embedding sums token + position + segment embeddings.
        # hidden shape: (batch, seq_len, hidden_dim).
        hidden = model.embedding(x, segment_info)
        for i, block in enumerate(model.transformer_blocks):
            if i >= layer:  # stop before block index `layer` (0 = embedding only)
                break
            hidden = block.forward(hidden, mask)

    # .detach() severs any residual autograd graph before numpy conversion.
    hidden = hidden.detach()
    if pool == "mean":
        # Average across dim=1 (the seq_len axis). dim=0 is batch, dim=2 is
        # hidden. Result shape: (batch, hidden_dim). NOTE: this averages
        # over padding positions too - that's what the shipped
        # UsableTransformer does, and what C2 quantifies.
        return torch.mean(hidden, dim=1).numpy()
    if pool == "none":
        return hidden.numpy()
    raise ValueError(f"unknown pool mode {pool!r}")


def cosine_sim(a, b):
    """Cosine similarity of two vectors.

    cos(a, b) = <a, b> / (||a|| * ||b||). Range is [-1, 1]:
        +1  -> same direction (identical up to positive scale)
         0  -> orthogonal, no linear relationship
        -1  -> opposite direction.
    """
    # .reshape(-1) flattens any batch dim; .reshape(-1) + np.dot replicates an
    # inner product cleanly whether inputs are 1-D or row/column vectors.
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    # Add a tiny epsilon to the denominator to avoid div-by-zero on a zero
    # vector (shouldn't happen with trained embeddings, but belt + braces).
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pairwise_cosine(matrix):
    """Pairwise cosine similarity of rows. Returns (N, N) matrix.

    Trick: normalize every row to unit L2 length, then the dot product
    between any two rows equals their cosine similarity. One matmul replaces
    an O(N^2) Python loop.
    """
    x = np.asarray(matrix, dtype=np.float64)
    # np.linalg.norm with axis=1 computes L2 norm per row.
    # keepdims=True preserves the axis so the shape stays (N, 1), which
    # broadcasts against x of shape (N, D) in the division below.
    norms = np.linalg.norm(x, axis=1, keepdims=True)   # (N, 1)
    x_unit = x / (norms + 1e-12)                       # (N, D) unit-length rows
    # x_unit @ x_unit.T -> (N, D) @ (D, N) = (N, N). Each cell is the dot
    # product of two unit vectors, which equals their cosine similarity.
    return x_unit @ x_unit.T


def print_header(title, char="="):
    line = char * max(len(title), 70)
    print(f"\n{line}\n{title}\n{line}")


def print_subheader(title):
    print(f"\n--- {title} " + "-" * max(1, 66 - len(title)))


def print_similarity_matrix(labels, sim, max_label_width=30, title=None):
    """Print a labeled cosine-similarity matrix. Good for N up to ~12."""
    if title:
        print_subheader(title)
    w = min(max_label_width, max(len(str(lbl)) for lbl in labels))
    header = " " * (w + 2) + "  ".join(f"{i:6d}" for i in range(len(labels)))
    print(header)
    for i, lbl in enumerate(labels):
        row = f"{str(lbl)[:w]:<{w}}  " + "  ".join(f"{sim[i, j]:+.3f}" for j in range(len(labels)))
        print(f"{i:2d} {row}")


def oov_tokens(vocab, instruction):
    """Return list of (token, is_oov) tuples for each whitespace-split token.

    Uses vocab.stoi directly (string -> index), treating unk_index (1) as OOV.
    This is intentionally simpler than vocab.to_seq so we can report
    per-token status without preprocessing side effects.
    """
    unk = vocab.unk_index
    return [(t, vocab.stoi.get(t, unk) == unk) for t in instruction.split(" ")]


def print_oov_report(vocab, instructions, label="instructions"):
    """Print a compact per-instruction OOV summary.

    PalmTree's 6631-token vocab was trained on Coreutils/Binutils (GCC/Clang).
    Libc symbol names, unusual literals, and many SIMD mnemonics are OOV and
    collapse to <unk>, producing identical embeddings across all such
    instructions. Surfacing this makes experiment findings interpretable.

    Returns True if any token is OOV.
    """
    any_oov = False
    print(f"  OOV check on {label}:")
    for ins in instructions:
        tokens = oov_tokens(vocab, ins)
        oovs = [t for t, is_oov in tokens if is_oov]
        if oovs:
            any_oov = True
            print(f"    [OOV: {', '.join(oovs):30s}] {ins}")
        else:
            print(f"    [ok]                             {ins}")
    if not any_oov:
        print("    (all tokens in vocab)")
    return any_oov


def top_k_nearest(query, matrix, labels, k=5, exclude_self=True):
    """Return top-k (label, cosine_sim) pairs from `matrix` nearest to `query`.

    If `query` is itself a row of `matrix`, pass exclude_self=True to skip it.
    """
    query = np.asarray(query, dtype=np.float64).reshape(-1)           # (D,)
    matrix = np.asarray(matrix, dtype=np.float64)                     # (N, D)
    # Normalize query and every row of matrix, same trick as pairwise_cosine.
    q_norm = query / (np.linalg.norm(query) + 1e-12)
    m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    # (N, D) @ (D,) = (N,). sims[i] is cos(query, matrix[i]).
    sims = m_norms @ q_norm
    # np.argsort returns indices that would sort ascending; negating gives
    # descending-by-similarity order (most similar first).
    order = np.argsort(-sims)
    out = []
    for idx in order:
        # np.allclose treats two arrays as equal within floating tolerance.
        # Used here to skip `query` if it happens to be a row of `matrix`.
        if exclude_self and np.allclose(matrix[idx], query):
            continue
        out.append((labels[idx], float(sims[idx])))
        if len(out) == k:
            break
    return out
