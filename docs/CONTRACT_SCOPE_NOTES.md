# Contract scope: the six contracts off the stress test (direction B) — results & writing notes

**Executed 2026-09-28.** Data: `results/contract_scope.json`; runner:
`a2i/contract_scope.py` (6 contracts × 6 non-D2 scenarios = 36
conditions, 0.5B, greedy, identical generation path to
s4_reversion/contract_gap).

## 0. Deployment and anchor

fp16 on GPU; the S1 column anchors against the round-3 fp32-CPU rows:
**ALL SIX bit-identical (≤1e-12)**. (E's stored value is exactly
0.9997969775 --- displayed as 1.000 at three decimals; both runs
reproduce it bit-exactly.)

## 1. Result

| scen | S1 | S2 | S3 | S4 | S3P | S4M |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 1.000 | 1.000 | 1.000 | 1.000 | **0.492** | 1.000 |
| A2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| A3 | 0.998 | 0.998 | **0.663** | 1.000 | 0.770 | 1.000 |
| B2 | 1.000 | 1.000 | 0.922 | 1.000 | 1.000 | 1.000 |
| B3 | 0.996 | 0.996 | 0.829 | 0.996 | 0.925 | 1.000 |
| E  | 0.9998 | 0.9998 | 0.9998 | 0.9998 | **0.338** | 0.9998 |

- S1/S2 (and S4/S4M): deployed level or ceiling everywhere (S4M lifts
  B3 0.996 → 1.000). No information-level boundary exists off D2.
- **S3 degrades 3/6** (A3 0.663, B3 0.829, B2 0.922): the per-family
  source breakdown that separates concurrent twins also separates the
  multiple sources of single-incident multi-source campaigns (A3's
  botnet); dims show [fam,dst]↔[fam,src] switching.
- **S3P unstable on 4/6** (E 0.338, A1 0.492, A3 0.770, B3 0.925):
  three distinct dim sets per run --- policy-churn fragmentation, the
  GLM failure mode induced in the 0.5B by a contract change.

## 2. Interpretation (pre-registered branch "contract is
scenario-dependent")

The S3 optimum does NOT transfer as a universal design rule. Combined
with direction A (S3 recovers 37% of the mixed-regime oracle gap but
churns under noise) the scoped claim is: the information contract is
the binding lever, but no single static contract is optimal across
regimes --- S1 is the benign-stream optimum, S3 the concurrency
optimum that pays for it elsewhere, S3P a destabilizer. Contract
changes are never local: they retune selection stream-wide. Deployment
consequence: runtime contract validation (stress-instance check) with
safe degradation to the incumbent contract. This also confirms, from
the other side, that D2 is the necessary discriminative instance for
the contract question.

What NOT to claim: "more information hurts" (S4/S4M are at ceiling off
D2 --- the D2 reversion is concurrency-specific); "S3 is a trap" (it
is the concurrency optimum; it just does not transfer).

## 3. Feeds into (stage 5)

- Results subsection \label{sec:contract-scope} + tab:scope (after
  §info; resolves the dangling forward reference).
- Cross-referenced by the Discussion oracle-gap paragraph.
- Sharpens the five-point chain's point 3 wording (contract decides;
  optimum is regime-dependent) --- abstract point (3) unchanged
  (D2-scoped statement already).
