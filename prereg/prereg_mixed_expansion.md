# Pre-registration: mixed-regime instance expansion (n=8 → 20)

**Date fixed: 2026-09-21 (before any expanded instance is generated or run).**
Status: parameters below are frozen; execution is a single batch scheduled
after the stage-1 Suricata replay conclusions, per the journal plan
(`PLAN.md`, stage 4). No instance may be added, removed, or re-parameterized
after any expanded result has been observed. The paper will describe this
analysis as pre-specified.

## Motivation

The regime subgroup analysis (round3_stats.subgroup, tab:subgroup) rests on
n=8 mixed-regime instances; advisor feedback (feedback/omar/next_step.md,
"One Statistical Fix") requires a pre-specified expansion to ~20. The
primary confirmatory test of the paper (adaptive vs. incumbent, Wilcoxon,
p=0.037) is locked and is **not** touched by this expansion; it addresses
only the mechanistic mixed-regime subgroup.

## Exact generator specification

New instances are P25–P36, produced by the same parameterized generator and
the same seed (17) as P01–P24 (round3.build_parametric semantics):

- day ∈ {tuesday, wednesday, thursday, friday};
- concurrency injection (twin) **on** for every new instance: the day's
  volumetrically largest attack family receives a concurrent same-family
  campaign from a second attacker (205.174.165.70), frac=0.5, identical to
  the round-3 twin rule;
- three new twin configurations, chosen from the *already-used* grid values
  only (ratios {1,4,16}; caps {750,1000,1500}; jitter {0.05,0.10}), filling
  the low-noise corner the existing twin set (ratios 8, 16) does not cover:

| config | ratio | cap | jitter | twin | instances |
| ------ | ----- | --- | ------ | ---- | --------- |
| C7     | 1     | 1500 | 0.05  | yes  | P25–P28 (tue/wed/thu/fri) |
| C8     | 4     | 1000 | 0.10  | yes  | P29–P32 (tue/wed/thu/fri) |
| C9     | 16    | 750  | 0.05  | yes  | P33–P36 (tue/wed/thu/fri) |

Mixed-regime set after expansion: {P05, P06, P11, P12, P17, P18, P23, P24}
(the round-3 twin instances) ∪ {P25–P36} = **n=20**. Single-regime set
unchanged (n=16). Total suite: P01–P36.

## Execution protocol (single batch)

1. Generate P25–P36 and run every arm on them in one pass: static
   incumbent, hindsight-best fixed (full DIM_SETS sweep), Julisch (+ its
   hindsight grid), both embedding arms, PSH hybrid, oracle, and the
   0.5B LLM hybrid — the identical free_arms composition of
   round3.py's parametric part, same seeds, same delta 300 s windows.
2. Append rows with part="parametric" to results/round3_expansion.json
   (P01–P24 results are not re-run and not modified).
3. Analysis, fixed in advance: round3_stats.subgroup_expanded — the same
   per-arm means, the same four paired Wilcoxon comparisons with bootstrap
   CIs, and the same two Mann-Whitney regime×arm interaction tests as the
   existing subgroup(), computed once on the n=20 mixed set. No additional
   post-hoc comparisons are pre-approved; anything exploratory found later
   must be labeled exploratory in the paper.

## Interpretation contract (written before the run)

Any outcome is reportable: the expansion narrows CIs on whichever effect
direction the data take. The claim structure of the paper does not depend
on the sign of the mixed-regime point estimates; it depends on (i) the
locked primary test and (ii) the oracle gap remaining positive. If the
expanded mixed-regime estimates reverse the n=8 direction, the paper
reports the reversal as the expanded estimate and treats the n=8 result as
the underpowered first look, with both disclosed.
