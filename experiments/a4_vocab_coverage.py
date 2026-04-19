"""A4 - Vocabulary Coverage.

Question
--------
PalmTree's vocabulary has 6,631 tokens built from Binutils/Coreutils compiled
with GCC/Clang. What fraction of tokens in a realistic instruction stream
falls outside this vocabulary (-> <unk>)? Which categories are least
covered?

Procedure
---------
No external binary is required. We define a synthetic corpus of instructions
grouped by expected coverage tier:

    1) CORE          - widely-used x86 in Coreutils: mov, add, jmp, call, ...
    2) SYSCALL_PRIV  - system / privileged instructions rarely in userspace:
                       syscall, sysret, wrmsr, rdmsr, cli, sti, int 0x80
    3) SIMD          - SSE/AVX vector instructions: movaps, vpxor, vmovdqa
    4) STRING_OP     - x86 string primitives: rep movsb, scasb, lodsq
    5) LIBC_CALLEES  - common C stdlib symbols used as `call <fn>` targets
    6) RAW_LITERALS  - unnormalized hex constants of various widths

For each tier we tokenize every instruction, look up each token in
vocab.stoi, and count how many map to unk_index (1).

What the result reveals - concrete consequences
-------------------------------------------------
- **CORE tier ~0% OOV (expected)**: basic userspace x86 is fully covered.
  Any nonzero CORE OOV means the vocab file is wrong or truncated - stop
  and re-verify the pickle load.
- **SIMD/AVX tier 30-60% OOV (observed ~32%)**: every `ymm*` and many
  `vpXXX` / `vmovXXX` mnemonics are missing. Practical consequence:
  performance-critical code (crypto, media, ML kernels, modern `memcpy`
  that goes through AVX) cannot be meaningfully represented. A
  function-level embedding of an AVX-heavy binary will be dominated by
  `<unk>` tokens and collapse to an uninformative vector. If your thesis
  evaluates PalmTree on OpenSSL or a video codec, your numbers will look
  worse than PalmTree's real capability simply because of vocabulary
  mismatch - not a model-quality issue.
- **SYSCALL_PRIV tier ~50%+ OOV (observed ~56%)**: kernel code, bootloaders,
  hypervisors, and anything using privileged instructions is out of scope.
  Consequence: PalmTree is NOT the right embedding for a kernel-fuzzing or
  syscall-clustering task without vocab extension or retraining.
- **LIBC_CALLEES ~50% OOV (every function name misses)**: the only token
  that distinguishes `call memcpy` from `call malloc` is the callee - and
  it's unk. Consequence (also surfaced in A2): your pipeline must rename
  libc targets to `symbol` (or richer normalization) BEFORE encoding or
  you lose all library-call distinction.
- **RAW_LITERALS ~30% OOV (by design)**: large hex constants collapse to
  `<unk>` because PalmTree expects a preprocessor (Binary Ninja in the
  paper) to have already mapped them to `address` / `string`. If you're
  feeding raw objdump output, you're skipping this step and silently
  degrading the embedding. The fix is a ~10-line normalization pass that
  maps constants wider than 6 hex digits (but narrower than 15) to
  `address` when absent from the symbol table, `symbol` when present.
- **Top OOV token list (printed at the end)**: tells you concretely WHICH
  tokens are missing. If a single OOV token dominates (e.g. `ymm0`
  appearing 5 times), you may be able to plug the gap with a targeted
  vocab extension rather than a full retrain. If the list is scattered
  across hundreds of different tokens, retrain or swap models.
- **Total OOV rate**: one number to report in a thesis chapter when arguing
  "PalmTree's vocabulary is shaped like Coreutils". A rate >20% on a
  non-Coreutils corpus is a defensible concrete claim.
"""

from __future__ import annotations

from collections import Counter

from . import _common as c


TIERS = {
    "CORE (Coreutils-like userspace)": [
        "mov rax rbx",
        "add rax rcx",
        "sub rdx rbx",
        "cmp rax 0x0",
        "jmp address",
        "call symbol",
        "ret",
        "push rbp",
        "pop rbp",
        "lea rax [ rbx + rcx * 0x4 ]",
        "xor rax rax",
        "test rax rax",
    ],
    "SYSCALL_PRIV (kernel / privileged)": [
        "syscall",
        "sysret",
        "sysenter",
        "sysexit",
        "wrmsr",
        "rdmsr",
        "cli",
        "sti",
        "int 0x80",
        "iret",
        "lgdt [ rax ]",
        "swapgs",
    ],
    "SIMD (SSE / AVX vector)": [
        "movaps xmm0 xmm1",
        "movdqa xmm0 [ rbp - 0x10 ]",
        "vpxor ymm0 ymm0 ymm0",
        "vmovdqa ymm0 [ rax ]",
        "vbroadcastsd ymm0 xmm1",
        "pshufb xmm0 xmm1",
        "addss xmm0 xmm1",
    ],
    "STRING_OP (rep / string primitives)": [
        "rep movsb",
        "rep stosb",
        "rep scasb",
        "lodsq",
        "stosq",
        "repne scasb",
    ],
    "LIBC_CALLEES (raw symbol names)": [
        "call malloc",
        "call free",
        "call memcpy",
        "call memset",
        "call strcmp",
        "call strlen",
        "call printf",
        "call fprintf",
        "call fopen",
    ],
    "RAW_LITERALS (un-normalized hex)": [
        "mov rax 0xdeadbeef",
        "mov rbx 0xcafebabe",
        "add rax 0x123456789abc",
        "mov [ 0x401020 ] rax",
        "call 0x401020",
    ],
}


def run(model, vocab):
    c.print_header("A4 - Vocabulary coverage across instruction tiers")
    # vocab.stoi is the string-to-index dict. Its length == vocab size.
    # vocab.unk_index (typically 1) is the fallback for unknown tokens.
    print(f"Vocab size: {len(vocab.stoi)} tokens (unk_index={vocab.unk_index})")

    totals_tokens = 0
    totals_oov = 0
    tier_summary = []
    # Counter accumulates how often each OOV token appears across ALL tiers.
    # Useful for the "top offenders" summary at the end.
    all_oov_tokens = Counter()

    for tier, instructions in TIERS.items():
        c.print_subheader(tier)
        tier_tok = 0
        tier_oov = 0
        for ins in instructions:
            # c.oov_tokens splits on whitespace and looks up each token in
            # vocab.stoi; returns list[(token, is_oov_bool)].
            toks = c.oov_tokens(vocab, ins)
            # Pull out just the tokens that hit <unk>.
            oov = [t for t, is_oov in toks if is_oov]
            tier_tok += len(toks)
            tier_oov += len(oov)
            totals_tokens += len(toks)
            totals_oov += len(oov)
            # Counter.update(iterable) increments counts for every element.
            # For an OOV token that appears twice in this instruction, the
            # counter gets +2 for it - which is what we want for "top
            # offenders" ranking.
            all_oov_tokens.update(oov)
            if oov:
                print(f"    [{len(oov)}/{len(toks)} OOV: {', '.join(oov)}]  {ins}")
            else:
                print(f"    [0/{len(toks)} OOV]                          {ins}")
        # Guard against zero-division if a tier is empty (shouldn't happen,
        # but keeps the function total-safe if someone edits TIERS later).
        rate = tier_oov / tier_tok if tier_tok else 0.0
        tier_summary.append((tier, tier_tok, tier_oov, rate))
        print(f"  tier OOV rate: {tier_oov}/{tier_tok} = {rate:.1%}")

    c.print_subheader("Per-tier summary")
    print(f"  {'tier':<40s}  {'tokens':>8s}  {'oov':>6s}  {'rate':>6s}")
    for tier, tok, oov, rate in tier_summary:
        print(f"  {tier[:40]:<40s}  {tok:>8d}  {oov:>6d}  {rate:>6.1%}")
    print(f"  {'TOTAL':<40s}  {totals_tokens:>8d}  {totals_oov:>6d}  "
          f"{totals_oov / totals_tokens:>6.1%}")

    if all_oov_tokens:
        c.print_subheader("Top OOV tokens (appearances)")
        for tok, count in all_oov_tokens.most_common(15):
            print(f"    {count:3d}x  {tok!r}")
