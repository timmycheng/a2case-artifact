# Contract design and the mixed-regime oracle gap (direction A) — results & writing notes

**Executed 2026-09-28.** Data: `results/contract_gap.json`; runner:
`a2i/contract_gap.py` (S3 vs S1, the 8 round-3 twin instances P05/P06/
P11/P12/P17/P18/P23/P24, rebuilt bit-identically, seed 17; 0.5B,
greedy, same prompt/validation/fallback path as s4_reversion).

## 0. Deployment and anchor chain (provenance)

Run on float16 GPU (the scale ablation's established bit-equivalent of
the paper's fp32-CPU deployment for this model). The S1 condition is
the in-batch anchor against the round-3 CPU rows: **7/8 instances
bit-identical (≤1e-12)**; P23 reproduces 0.6451 under BOTH fp16-GPU
and a dedicated fp32-CPU re-run, vs the stored round-3 row's 0.6458
(Δ=0.0007, one borderline window; the stored row is the deviating
side, the re-run value is reported). Oracle / best-fixed / incumbent
joined from the round-3 rows (deterministic construction).

## 1. Result (final, P23 CPU-patched)

| inst | day | S1 | S3 | d | oracle | best-fixed |
| --- | --- | --- | --- | --- | --- | --- |
| P05 | tue | 0.6725 | **1.0000** | +0.3275 | 1.0000 | 1.0000 |
| P06 | tue | 0.6717 | 0.6358 | −0.0359 | 1.0000 | 1.0000 |
| P11 | wed | 0.8665 | 0.7761 | −0.0904 | 0.9678 | 1.0000 |
| P12 | wed | 0.8700 | 0.7170 | −0.1529 | 1.0000 | 1.0000 |
| P17 | thu | 0.5461 | **0.9785** | +0.4323 | 0.9950 | 0.9997 |
| P18 | thu | 0.4838 | 0.3659 | −0.1179 | 1.0000 | 0.9999 |
| P23 | fri | 0.6451 | **0.9195** | +0.2744 | 1.0000 | 0.8532 |
| P24 | fri | 0.7070 | **1.0000** | +0.2930 | 1.0000 | 0.8533 |

Means: S1 0.6828 → S3 0.7991 (oracle 0.9954, best-fixed 0.9633).
Gap to oracle: 0.3125 → 0.1962; **recovered fraction 0.372**.
S3−S1 = +0.1163, Wilcoxon p=0.3125, 95% CI [−0.039, +0.269], 4/8 wins.
(Note S3 exceeds *best-fixed* on P23/P24 where no fixed config is
perfect — the two instances that motivated the open problem.)

## 2. Mechanism (dims histograms)

- S1 everywhere: constant [family,dst] (P23/P24 add [src,dst] from 10
  JSON-fallbacks each) — the family-keyed over-merge of the paper.
- S3 is the only contract under which the model issues source-splitting
  [family,src] — exactly the D2 mechanism, now on the naturally mixed
  instances. Where twins dominate the window content (P05, P17, P23,
  P24 — the 8×-noise configs), it lands near-oracle (three ≥0.98,
  two at 1.000).
- Where background noise dominates (P06/P12/P18 are 16×-ratio configs;
  P11 the wednesday 8× exception), the same field induces
  [fam,dst]↔[fam,src] switching churn that fragments about as often as
  it repairs (losses −0.04..−0.15).

## 3. Interpretation (pre-registered frame, branch "partial recovery")

The mixed-regime oracle gap is **partially contractual**: the single
summary field that solves D2 recovers 37% of the mean gap on the hard
instances — fully on the twin-dominated ones, including both instances
where no fixed configuration is perfect — without touching model,
architecture, or training. The recovered share is heterogeneous and
noise-dependent: under heavy background the enriched contract
destabilizes selection (churn), a failure mode the direction-B scope
study confirms is not D2-specific. The **residual gap (S3 0.799 vs
oracle 0.995)** is therefore not contract-absence but noise-diluted
selection — the sharpened open problem is a contract that signals
concurrency without destabilizing selection under noise (contract
validation at run time, regime-conditional contracts).

What NOT to claim: "S3 closes the gap" (it recovers 37%, p=0.31,
n=8); "the gap is contractual" (only partially — say "partially");
significance language (mechanistic study, n=8; the confirmatory test
of the paper remains the locked adaptive-vs-incumbent comparison).

## 4. Feeds into (stage 5)

- Discussion subsection "Contract Design and the Oracle Gap".
- Cross-reference with direction B (contract scope): same field that
  repairs mixed instances degrades benign ones (A3 0.663) — the
  contract is regime-dependent, which is the runtime-validation
  design argument.

## n=20 pre-registered replication + seeds (2026-10-08, P0-R of the Omar brief)

Harness: a2i/contract_gap_n20.py; prereg: prereg_contract_n20.md (frozen
before any run). results/contract_gap_n20.json. 40 mixed rows + 14 seed
rows. All 20 S1 anchors bit-identical to stored round-3 rows except P23
(GPU fp16 flips one borderline window, 0.6451 vs 0.6458 — same instance
and magnitude as the original n=8 study; both P23 levels re-run CPU/fp32
per protocol).

Seed replication (expectation confirmed):
- D2: S3 = 1.000 on seeds {17,29,43}; S1 = 0.8856 on all three (identical).
- P12 (noise-dominated, S3 loses at seed 17): S1 0.857/0.865, S3 0.628/0.689
  at seeds {29,43} — loss direction stable, magnitude varies ~0.06.
- P24 (twin-dominated, S3 wins at seed 17): S1 0.675/0.678, S3 0.832/0.897
  at seeds {29,43} — win direction stable, magnitude varies ~0.07.
Greedy decode is deterministic given the stream; spread = stream-seed only.

Mechanism-trace caveat (repaired post hoc, see a2i/trace_patch.py): the
in-run twin counter matched dataset labels ('FTP-Patator') against catalog
families ('BRUTE-FTP') so stored counts were 0; windows are deterministic
functions of the stream, so counts were recomputed from rebuilt instances
by gt_incident pair {i, i+'-twin'} — no model outputs touched. Under the
preregistered >=0.5 twin-pair-share rule, 73 of ~2,140 S3 windows are
twin-dominated (concentrated in the low-noise C7/C8 expansion configs);
final split counts in the analyze log.
