"""
Run PalmTree pre-trained model on Apple Silicon (CPU mode).

Handles:
- bert_pytorch -> palmtree module redirect (pickle compat)
- CUDA -> CPU device mapping
- Deprecated scipy imports

Usage:
    source .venv/bin/activate
    python run_palmtree.py
"""

import sys
import os

# --- Module shim: pickle expects 'bert_pytorch.*', source lives at 'palmtree.*' ---
# Add src/ to path so 'palmtree' package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# Add pre-trained_model/ so vocab.py and config.py are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pre-trained_model"))

import palmtree as _palmtree_pkg
sys.modules["bert_pytorch"] = _palmtree_pkg

# Register all submodules that the pickle references
import palmtree.dataset.vocab
import palmtree.model.bert
import palmtree.model.transformer
import palmtree.model.attention.single
import palmtree.model.attention.multi_head
import palmtree.model.embedding.bert
import palmtree.model.embedding.position
import palmtree.model.embedding.segment
import palmtree.model.embedding.token
import palmtree.model.utils.feed_forward
import palmtree.model.utils.gelu
import palmtree.model.utils.layer_norm
import palmtree.model.utils.sublayer

sys.modules["bert_pytorch.dataset"] = palmtree.dataset
sys.modules["bert_pytorch.dataset.vocab"] = palmtree.dataset.vocab
sys.modules["bert_pytorch.model"] = palmtree.model
sys.modules["bert_pytorch.model.bert"] = palmtree.model.bert
sys.modules["bert_pytorch.model.transformer"] = palmtree.model.transformer
sys.modules["bert_pytorch.model.attention"] = palmtree.model.attention
sys.modules["bert_pytorch.model.attention.single"] = palmtree.model.attention.single
sys.modules["bert_pytorch.model.attention.multi_head"] = palmtree.model.attention.multi_head
sys.modules["bert_pytorch.model.embedding"] = palmtree.model.embedding
sys.modules["bert_pytorch.model.embedding.bert"] = palmtree.model.embedding.bert
sys.modules["bert_pytorch.model.embedding.position"] = palmtree.model.embedding.position
sys.modules["bert_pytorch.model.embedding.segment"] = palmtree.model.embedding.segment
sys.modules["bert_pytorch.model.embedding.token"] = palmtree.model.embedding.token
sys.modules["bert_pytorch.model.utils"] = palmtree.model.utils
sys.modules["bert_pytorch.model.utils.feed_forward"] = palmtree.model.utils.feed_forward
sys.modules["bert_pytorch.model.utils.gelu"] = palmtree.model.utils.gelu
sys.modules["bert_pytorch.model.utils.layer_norm"] = palmtree.model.utils.layer_norm
sys.modules["bert_pytorch.model.utils.sublayer"] = palmtree.model.utils.sublayer

# --- Now load and use the model ---
import torch
import numpy as np
import pickle

MODEL_DIR = os.path.join(os.path.dirname(__file__), "pre-trained_model", "palmtree")
MODEL_PATH = os.path.join(MODEL_DIR, "transformer.ep19")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab")

SEQ_LEN = 20  # max tokens per instruction (from training config)


def load_model():
    """Load pre-trained PalmTree model and vocab onto CPU."""
    print(f"Loading vocab from {VOCAB_PATH}")
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    print(f"  Vocab size: {len(vocab)}")

    print(f"Loading model from {MODEL_PATH}")
    model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.eval()
    print(f"  Model loaded (CPU mode)")
    return model, vocab


def encode(model, vocab, instructions):
    """
    Encode a list of assembly instructions into embeddings.

    Args:
        model: loaded PalmTree BERT model
        vocab: loaded WordVocab
        instructions: list of strings, e.g. ["mov rbp rdi", "call memcpy"]
                      tokens must be space-separated

    Returns:
        numpy array of shape (len(instructions), 128)
    """
    segment_label = []
    sequence = []
    for ins in instructions:
        tokens = ins.split(" ")
        # +2 for <sos> and <eos> tokens
        label = [1] * (len(tokens) + 2)
        seq = vocab.to_seq(ins)
        seq = [3] + seq + [2]  # 3=<sos>, 2=<eos>

        # Pad or truncate to SEQ_LEN
        if len(label) > SEQ_LEN:
            segment_label.append(label[:SEQ_LEN])
        else:
            segment_label.append(label + [0] * (SEQ_LEN - len(label)))
        if len(seq) > SEQ_LEN:
            sequence.append(seq[:SEQ_LEN])
        else:
            sequence.append(seq + [0] * (SEQ_LEN - len(seq)))

    segment_label = torch.LongTensor(segment_label)
    sequence = torch.LongTensor(sequence)

    with torch.no_grad():
        encoded = model.forward(sequence, segment_label)
        # Mean-pool across token dimension -> one vector per instruction
        result = torch.mean(encoded, dim=1)

    return result.numpy()


def main():
    model, vocab = load_model()

    # Example: x86 assembly instructions (space-separated tokens)
    instructions = [
        "mov rbp rdi",
        "mov ebx 0x1",
        "mov rdx rbx",
        "call memcpy",
        "mov [ rcx + rbx ] 0x0",
        "mov rcx rax",
        "mov [ rax ] 0x2e",
    ]

    print(f"\nEncoding {len(instructions)} instructions...")
    embeddings = encode(model, vocab, instructions)

    print(f"Output shape: {embeddings.shape}")
    print(f"  -> {len(instructions)} instructions x {embeddings.shape[1]}-dim embeddings\n")

    # Show per-instruction embedding norms (sanity check)
    for ins, emb in zip(instructions, embeddings):
        norm = np.linalg.norm(emb)
        print(f"  {ins:30s}  ||emb|| = {norm:.4f}")

    # Cosine similarity between first two instructions (both 'mov' variants)
    from numpy.linalg import norm
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    print(f"\nCosine similarity('mov rbp rdi', 'mov ebx 0x1') = {cos_sim:.4f}")

    cos_sim2 = np.dot(embeddings[0], embeddings[3]) / (norm(embeddings[0]) * norm(embeddings[3]))
    print(f"Cosine similarity('mov rbp rdi', 'call memcpy')  = {cos_sim2:.4f}")


if __name__ == "__main__":
    main()
