# Pre-registration: S1-vs-S3 contract contrast on the full mixed-regime suite (n=20), with seed replication

**Date fixed: 2026-10-08 (before any n=20 contract run or seed-replication run).**
Status: everything below is frozen before execution. Advisor brief
(`feedback/omar/review-alerts-to-cases.md`, 2026-10-08, P0 "Run S3 (and S1
control) on the full mixed-regime n=20" and P1 "Multiple seeds on the
contract result") requires exactly this experiment and pre-authorizes both
outcomes: "If the recovery shrinks, report that; it is still a paper."
No instance, condition, or test may be added, removed, or re-parameterized
after any result below has been observed. The paper will describe this
analysis as pre-registered.

## Instances

The pre-registered mixed-regime suite (n=20), identical to
`prereg_mixed_expansion.md`:
{P05, P06, P11, P12, P17, P18, P23, P24} (round-3 twins) ∪ {P25–P36}
(expansion batch), built bit-identically via `round3.build_parametric(zf,
pid, seed=17)`. The n=8 results of the original contract-gap study
(`results/contract_gap.json`) are retained unchanged and reported beside
the n=20 numbers.

## Conditions and deployment

Two conditions only, as in the n=8 study: S1 (deployed contract; also the
reproducibility anchor) and S3 (the D2 sharp-optimum contract). Model:
Qwen2.5-0.5B-Instruct, greedy, max_new_tokens 96, identical prompt/schema/
fallback semantics — only the summary block differs (levels via
`classics.stream_summary`). Deployment: float16 on GPU; the S1 arm is
re-run in-batch as an anchor and **must** match the stored round-3 rows
(`hybrid_llm_0.5B` in `results/round3_local.json` for P05–P24 and
`results/round3_expansion.json` for P25–P36) to 1e-12 per instance; any
diverging instance is re-run on CPU/float32 for **both** levels and the
CPU rows replace the GPU rows for that instance (patch mode). Oracle /
best-fixed / incumbent reference rows are not re-run for the contrast.

## Primary statistics (fixed)

On the n=20 mixed suite, for the paired differences S3−S1 and oracle−S3:

- two-sided Wilcoxon signed-rank on the instance-level ARI differences,
  zero differences dropped, exact null distribution when no ties remain
  among the nonzero differences, normal approximation otherwise (the
  paper's stated convention, §6.3);
- percentile bootstrap 95% CI of the mean difference, 20,000 resamples,
  `numpy.random.default_rng(17)`;
- win/tie/loss counts; gap-recovery fraction
  `((oracle−S1) − (oracle−S3)) / (oracle−S1)` on per-arm means.

These are computed once, on the full n=20, and reported beside the
retained n=8 numbers. Nothing else is pre-approved as confirmatory;
anything found later is labeled exploratory in the paper.

## Mechanism counts (fixed rule, written before the run)

Per window, the **twin-family fraction** is the share of the window's
alerts whose `family` equals the day's injected twin family
(tuesday FTP-Patator, wednesday DoS Hulk, thursday Web Attack — Brute
Force, friday PortScan — this counts both campaigns of the twin pair,
the original and the injected 205.174.165.70 source). A window is
**twin-dominated** iff this fraction is ≥ 0.5, else **noise-dominated**.
The per-window policy trace stores, for every window: chosen dims,
chosen window_min, fallback flag, window alert count, twin-family count,
and twin-source count (so the split is recomputable under either
definition; the ≥0.5 twin-family rule above is the primary, frozen one).
Reported: counts of windows emitting `family,src` vs `family,dst` vs any
other dims, split by twin-dominated vs noise-dominated, for S3 on the
n=20 suite.

## Seed replication (fixed)

- D2: `scenarios.build_twins(zf, seed ∈ {17, 29, 43})`, arms S1 and S3.
- Two mixed instances chosen for contrast before the run: P24 (an S3
  winner at seed 17, twin-dominated) and P12 (an S3 loser at seed 17,
  noise-dominated), `build_parametric(seed ∈ {17, 29, 43})`; seed 17
  comes from the main batch above, seeds 29/43 are fresh runs.
- Deployment identical to the main batch; greedy decoding is
  deterministic given the stream, so seed variation reflects the
  stream-construction seed only. Reported: per-seed ARI per arm and the
  spread; expectation (falsifiable): S3 = 1.000 on D2 replicates across
  seeds, and the P24/P12 gain/loss direction is stable.

## Complementary metric (fixed definition)

B-Cubed precision/recall/F1 with multiplicity, computed on the
**attack-alert restriction only** (identical restriction to the paper's
ARI/purity metrics; background alerts excluded). Computed in-batch for
S1 and S3 on the n=20 suite, and deterministically (no LLM) for the
incumbent, hindsight-best fixed, PSH, and oracle arms on the same 20
instances (the incumbent-sensitivity batch). ARI remains the primary
metric; B-Cubed is reported as the complementary one.

## Interpretation contract (written before the run)

Any outcome is reportable. If the n=20 recovery fraction is smaller than
the n=8 value (37%), the paper reports the shrinkage and treats n=20 as
the pre-registered estimate with n=8 as the original look — both
disclosed, per the advisor brief. The paper's claim structure does not
depend on the magnitude of the recovery; it depends on the oracle-gap and
interaction results, which are untouched by this experiment.
