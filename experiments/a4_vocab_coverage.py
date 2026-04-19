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

What the result reveals
-----------------------
- A high OOV rate on CORE would invalidate the model's basic claim to
  represent x86 semantics. (Expected: ~0% OOV.)
- High OOV rates on SIMD and SYSCALL_PRIV tell you which production binary
  classes PalmTree cannot represent meaningfully.
- LIBC_CALLEES being almost entirely OOV demonstrates the practical
  limitation of treating callee symbols as raw tokens without a
  symbol-table-driven normalization pass.
- RAW_LITERALS being OOV is actually BY DESIGN - PalmTree's intended
  pre-processing maps large hex to `address`. If you skip that step, every
  literal is a fresh unk.
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
    print(f"Vocab size: {len(vocab.stoi)} tokens (unk_index={vocab.unk_index})")

    totals_tokens = 0
    totals_oov = 0
    tier_summary = []
    all_oov_tokens = Counter()

    for tier, instructions in TIERS.items():
        c.print_subheader(tier)
        tier_tok = 0
        tier_oov = 0
        for ins in instructions:
            toks = c.oov_tokens(vocab, ins)
            oov = [t for t, is_oov in toks if is_oov]
            tier_tok += len(toks)
            tier_oov += len(oov)
            totals_tokens += len(toks)
            totals_oov += len(oov)
            all_oov_tokens.update(oov)
            if oov:
                print(f"    [{len(oov)}/{len(toks)} OOV: {', '.join(oov)}]  {ins}")
            else:
                print(f"    [0/{len(toks)} OOV]                          {ins}")
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
