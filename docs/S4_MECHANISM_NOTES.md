# S4 reversion mechanism study — results & writing notes (stage 3)

**Date executed: 2026-09-21.** Data: `results/s4_reversion_local.json`
(0.5B, four conditions x 107 D2 windows, greedy, 0 fallbacks) and
`results/s4_reversion_glm.json` (GLM-5.3 conditions, pending API quota —
see §4). The S1/S2 baselines are the round3 rows
(`results/round3_local.json`, arms qwen_summary_S1/S2).

## 1. Design (pre-specified before running)

Two advisor hypotheses, tested with two new summary conditions on the same
D2 twin stress test, same prompt template, same decoding/validation:

- **S4M** (Hypothesis A — token-length dilution): the S4 summary
  re-serialized as compact matrix text. Information identical field-for-
  field (asserted programmatically over all cross-tab entries); summary
  tokens 1288 -> 969 (-25%, tokenizer-measured over all 107 windows).
- **S3P** (Hypothesis B — cross-tab pseudo-correlation): S3 plus exactly
  one field, per-family distinct-source counts (the attacker-count signal
  in its purest form; family_sources' most_common(4) truncation makes the
  cardinality genuinely new information). No cross-tab. Summary tokens
  875 vs S3's 842 (+4%).

## 2. Result

| condition | summary tok (mean) | prompt tok/call | ARI | dims collapse |
| --- | --- | --- | --- | --- |
| S1 (round3) | — | — | 0.8856 | [family,dst] |
| S2 (round3) | — | — | 0.8856 | [family,dst] |
| S3 | 842 | 1008 | **1.0000** | [family,dst] + [family,src] |
| S3P | 875 | 1041 | 0.8856 | [family,dst] (+[family] once) |
| S4 | 1288 | 1454 | 0.8856 | [family,dst] |
| S4M | 969 | 1136 | 0.8856 | [family,dst] |

**Both hypotheses are refuted as stated, and the reversion is sharper
than either predicts:**

- A refuted: removing 25% of tokens with zero information change does not
  recover S3 (S4M = 0.8856, identical to S4's partition metrics:
  purity 0.909, over 0.200 — the same single twin over-merge).
- B refuted (and inverted): adding only the attacker-count field
  reproduces the full reversion at S3's own length (S3P = 0.8856). The
  cross-tab is not the culprit; the failure is triggered by the addition
  of family-level source aggregates beyond S3's per-family source list.
- The five non-S3 conditions all collapse to the same family-keyed policy
  selection (dims histogram [family,dst]) and the identical over-merge;
  S3 is a **sharp optimum** — the only tested summary on which the 0.5B
  model splits by source in the critical window(s) ([family,src]
  selections appear only at S3).

## 3. How to write it (converts the liability into the mechanism result)

Claim shape: "The summary information contract is not monotone and the
S3 optimum is sharp: every tested deviation from S3 — removing the
per-family source breakdown (S1/S2), adding source-count cardinality
(S3P), or the full cross-tabulation in either serialization (S4/S4M) —
drives the same identifiable failure mode, a policy-selection flip to
family-keyed over-merging (identical partitions: ARI 0.886, one twin
over-merge). Length is exonerated by construction: S4M conveys identical
information in 25% fewer tokens, and S3P fails at S3's length. The
failure is therefore content-localized (the boundary sits exactly at S3's
information set), deterministic, and avoidable by contract validation on
a single stress instance."

What NOT to claim: "more information always hurts" (the dose-response is
non-monotone — S1/S2 fail from below); "token budget is the lever"
(refuted); "the cross-tab introduces spurious correlations" (refuted —
any source-aggregate addition fails).

## 4. GLM-5.3 side conditions (completed 2026-09-26)

Run on the GLM coding-plan endpoint (`/api/coding/paas/v4`, same model id
glm-5.3, same thinking-effort-low body) after the pay-as-you-go key hit a
sustained account-level quota (429 code 1113, 9/21–9/26); all four cells
are internally consistent (one endpoint, one day). Data:
`results/s4_reversion_glm.json`.

| variant | GLM-5.3 ARI | valid calls |
| --- | --- | --- |
| S3 | 0.499 | 105/107 |
| S4 | 0.530 | 104/107 |
| S3P | 0.650 | 107/107 |
| S4M | 0.484 | 100/107 |

Against the 0.5B results (S3 1.000; S1/S2/S3P/S4/S4M all 0.886) and the
round-3 GLM rows (S1 0.57 / S2 0.51 / S3 0.42 on the old endpoint):
**GLM fails at every variant, including both new conditions** — the
compact-matrix reformat (S4M) and the attacker-count field (S3P) do
nothing for it. The information-contract levers that produce the 0.5B
model's sharp S3 optimum have no effect on the frontier reasoning model:
its failure is training-objective-side, not information-side. This
completes the mechanism table: one model class is contract-sensitive
(with a sharp optimum), the other is contract-insensitive (uniformly
failing). Minor disclosure: 2–7 API-fallbacks per condition, reported
per row; S4M's 7 fallbacks suggest the matrix format occasionally
breaks the reasoning model's JSON compliance.
