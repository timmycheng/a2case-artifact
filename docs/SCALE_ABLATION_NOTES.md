# Scale-ablation writing package (stage 4) — results & drafts for stage 5

**Executed 2026-09-21/22; final curve complete including the deployment
exoneration anchor.** Data: `results/scale_ablation.json`
(`part="scale_ablation"`; 6 conditions x 7 original scenarios; 0 invalid
calls everywhere — 7B: 739/739 valid).

## 0. Result (complete)

| scenario | 0.5B fp32 cpu | 0.5B fp16 gpu | 1.5B gpu | 3B gpu | 3B offload | **7B offload** |
| --- | --- | --- | --- | --- | --- | --- |
| A1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.857 |
| A2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.742 |
| A3 | 0.998 | 0.998 | 1.000 | 1.000 | 1.000 | 0.729 |
| B2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.682 |
| B3 | 0.996 | 0.996 | 1.000 | 1.000 | 1.000 | 0.475 |
| E | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.731 |
| D2 | 0.886 | 0.886 | 0.886 | 0.886 | 0.886 | 0.621 |

- 0.5B fp32-CPU (paper configuration) and 0.5B fp16-GPU are identical on
  all seven scenarios — precision/device exonerated; the anchor bridges
  the published numbers to the curve.
- 1.5B and 3B equal or slightly improve the 0.5B level (A3, B3 -> 1.000);
  the D2 twin over-merge (0.886 under the S1 contract) is invariant
  0.5B-3B.
- 7B collapses on every scenario (mean 0.69 vs 0.99 for the rest), D2
  included (0.621).
- **Deployment exoneration**: 3B re-run under the identical hybrid
  GPU/CPU offload deployment as 7B reproduces pure-GPU 3B exactly (7/7).
  The 7B result is therefore attributable to the model, not to the
  offload kernel path. (The 7B deployment was forced: fp16 7B does not
  fit an 8 GB card; pure-CPU fp16 was ~2-3 min/call.)

## 0.1 Interpretation (the non-monotone dose-response)

The advisor's three-outcome frame anticipated monotone possibilities;
the data delivered a fourth, stronger shape: **the dose-response is
non-monotone — scale is flat and sufficient through 3B, then turns into
an active liability at 7B.** Claimable, in the paper's voice:

> Within the Qwen2.5 instruction family, policy-selection quality under
> the same contract is flat from 0.5B through 3B (the only scale effect
> is two scenarios moving from 0.998/0.996 to 1.000), and degrades
> sharply at 7B on every scenario, including the stress test. Scale
> within the instruction-tuned family is not the lever — and beyond the
> small-model regime it is a liability. Combined with the GLM-5.3
> result (a frontier reasoning model that fails at every summary
> level), neither more parameters nor a reasoning training objective
> improves policy selection; the binding constraint is the information
> contract, and the deployed small model sits at its optimum.

Caveats to state: single run per cell (greedy, deterministic — same
standard as the rest of the paper); 7B's absolute numbers are deployment-
conditioned only in the sense exonerated above; no claim about >7B.

## 1. Setup paragraph (model selection rationale) — draft

> **Models.** The scale ablation uses the Qwen2.5-Instruct family —
> 0.5B, 1.5B, 3B, and 7B parameters — so that every scale point shares
> the same pretraining data lineage, the same instruction-tuning
> paradigm, the same tokenizer and chat template, and the same deployment
> path through the policy contract. Across all scale points we hold
> fixed: the prompt (template and summary level S1), the decoding
> protocol (greedy, temperature 0), the JSON policy schema with static
> fallback, the micro-batch windowing (300 s), and the scenario grid
> (A1–E, B2/B3, and the D2 twin stress test). The 0.5B member is the
> paper's deployed configuration (float32, CPU); it is re-run in this
> grid as the bridging anchor, and additionally under float16 on GPU,
> which isolates the precision/device effect from the parameter-count
> effect (the two configurations produce identical partitions on all
> seven scenarios). The 1.5B and 3B members run float16 on a single 8 GB
> consumer GPU (RTX 4060 Ti); the 7B member does not fit in 8 GB and
> runs float16 with a hybrid GPU/CPU weight placement (accelerate
> device_map=auto) — the one disclosed deployment deviation, forced by
> memory, not by design. GLM-5.3 remains in the study as a training-objective
> contrast (reasoning-thinking vs. instruction-following), not as a
> scale endpoint: its API, thinking mode, and schema behavior differ by
> design, and treating it as "the large end of a scale curve" would
> confound scale with training objective.

## 2. Threats-to-validity responses — drafts

> *Why stop at 7B?* The paper's deployment claim concerns locally
> runnable small models; 7B is the largest member that a single
> consumer GPU class can host in common deployments, and the ablation's
> purpose is to separate scale from training objective, not to chart
> frontier scale. Beyond 7B the comparison stops being about the
> deployment configuration the paper argues for.

> *Why not Qwen3?* The Qwen3 family exposes a thinking-mode switch as
> part of the model interface. Enabling or disabling it changes the
> training objective that is active at inference — precisely the
> confound the same-family ablation is designed to hold fixed. Qwen2.5
> (pure instruction tuning, no thinking switch) is the newest family
> that keeps the controlled variable clean.

## 3. Interpretation contract (as written before the numbers landed)

The pre-registered frame anticipated three monotone outcomes; §0.1
records the fourth shape the data took (non-monotone: flat through 3B,
collapse at 7B), with the deployment exoneration that makes it
attributable. The outcome is reportable under the contract: any result
advances the scale/objective separation.

## 4. Mixed-regime expansion (executed)

Pre-registered 2026-09-21 (`prereg_mixed_expansion.md`); executed as a
single batch after the stage-1 conclusion. Output:
`results/round3_expansion.json`; analysis: `round3_stats.subgroup_expanded`
(fixed tests, n=20 mixed). **Outcome:** mixed-regime LLM−best-fixed
dmean = −0.270, p = 0.0001, 95% CI [−0.337, −0.204], 0/20 wins (n=8
gave −0.280, p = 0.008 — survived and tightened); LLM−incumbent −0.020,
p = 0.55; best-fixed mean 0.963 still imperfect vs oracle 0.996;
interactions MW p = 0.0012 / 1.2e-7; single-regime rows identical to the
paper's. Report as pre-specified.
