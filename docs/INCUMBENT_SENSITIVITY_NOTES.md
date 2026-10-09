# Incumbent sensitivity + candidate-space audit (2026-10-08)

Omar brief items executed here: P1 "Separate deployed incumbent from a weak
fixed rule" (family / family,dst columns, all 36 parametric instances),
P1 "Make the oracle and the policy space auditable" (D2 oracle trace +
14-candidate checks), P1-M B-Cubed complementary metric on mixed-20.
Harness: `a2i/incumbent_sensitivity.py` (deterministic, no GPU);
results in `results/incumbent_sensitivity.json`,
`results/oracle_d2_trace.json`. Prereg references:
`prereg_contract_n20.md` (BCubed definition frozen there).

## P1-R: adaptive (deployed 0.5B C1) vs family-keyed comparators

Identity checks: incumbent rerun and best-fixed rerun are BIT-IDENTICAL to
the stored round-3 rows on all 36 instances (instance reconstruction
verified, including the expansion builder for P25-P36).

| base | LLM − family | W/T/L | p (Wilcoxon, two-sided, zero-dropped) | bootstrap 95% CI (20k, rng17) |
|---|---|---|---|---|
| pooled-24 (locked suite) | −0.0149 | 0/19/5 | 0.0625 | [−0.0315, −0.0016] |
| single-16 | −0.0069 | 0/13/3 | 0.25 | [−0.0182, +0.0000] |
| mixed-20 | −0.0187 | 0/15/5 | 0.0625 | [−0.0398, −0.0018] |

- family means: single-16 **0.9999**, mixed-20 **0.7117**.
- **family,dst ≡ family on all 36 instances** (twin campaigns target the
  day's documented victims → family determines dst within these streams).
  Reported in the paper as a finding, not an assumption.
- Reading (both directions support the paper, as the brief predicted):
  single-regime +0.361 incumbent win = recovery of the family optimum
  (circularity); mixed-regime LLM (0.693) loses to family (0.712), both far
  below best-fixed 0.963 / oracle 0.996. Against a family-keyed incumbent
  there is no pooled adaptive win at all (0 wins in 24).

## Candidate-space audit (motivated by the boxed §7.6 definition)

The implemented static sweep / hindsight-best-fixed searches **7 dimension
sets at the deployed w=10 only**; the oracle trials all 14 (7 × w∈{5,10});
the 0.5B generator emitted w=10 on every traced call (n=20 contract batch;
historical grids did not log the timeout — paper wording scoped
accordingly).

D2 full 14-candidate static sweep (verify "only src,family is perfect"):

| dims | w=5 | w=10 |
|---|---|---|
| src,dst | 0.739 | 0.398 |
| src,dst,port | 0.739 | 0.398 |
| dst,port | 0.001 | 0.001 |
| family | 0.886 | 0.886 |
| family,dst | 0.886 | 0.886 |
| sid | 0.575 | 0.575 |
| **src,family** | **1.000** | **1.000** |

→ "only src,family reaches 1.00 on D2" holds across the FULL 14-candidate
space, not just the w=10 slice. Headline intact.

## P1-O: D2 oracle trace (results/oracle_d2_trace.json)

107 windows, final ARI 1.000. Run-length commitments:
w0-25 (src,dst,5) [global ARI undefined before w21 (no attack alerts),
1.000 from w21]; w26-30 (src,dst,dst_port,5); w31-87 (src,dst,5);
w88 (src,dst,10); w89-106 (src,dst,5).

**The oracle never picks src,family on D2.** It holds 1.000 via per-window
keying switches while static (src,dst,5) over the same stream = 0.739 →
per-window *selection* (not any single keying) reaches the menu ceiling.
In paper: Appendix B table `tab:oracle-d2` + one paragraph.

## P1-M: B-Cubed F1 (attack-alert restriction, frozen definition)

Final mixed-20 table (results/incumbent_sensitivity_analysis.log):

| arm | ARI | BCubed F1 |
|---|---|---|
| incumbent (src,dst,10) | 0.7132 | 0.8394 |
| family (= family,dst) | 0.7117 | 0.8608 |
| best-fixed (7@w10) | 0.9631 | 0.9731 |
| best-fixed (14 candidates) | 0.9631 | 0.9731 |
| PSH | 0.6788 | 0.8084 |
| oracle | 0.9964 | 0.9968 |

**bf14 definitional check: best-fixed over all 14 candidates changes 0/20
instances (dmean +0.0000)** — the 0.996-vs-0.963 headroom is insensitive
to the timeout slice; added to the boxed §7.6 paragraph as a sentence.
LLM S1/S3 BCubed fill from the contract batch once the GPU run completes
(partial: S1 0.853 n=14, S3 0.894 n=13).

## Paper deltas already applied (2026-10-08)

- §7.9: `tab:incsens` + "Incumbent sensitivity" paragraph.
- §7.6 boxed paragraph: static sweep scope corrected (7 sets @ w=10;
  generators + timeout; oracle 14; traced-choices claim).
- §4: `tab:defaults` shipped-defaults table (QRadar/Splunk/SecOps/Sentinel/
  Wazuh ↔ swept counterparts); vendor prose shortened.
- Appendix B: D2 oracle trace table + paragraph.
- tab:llm caption: GLM no-snapshot variation range (0.600 vs 0.575).
- tab:main caption: compression = full stream incl. FPs, with worked example.
- §5: mapping stated as a rule (incident id = (window, participants);
  infiltration exception as rule); signature-fidelity calibration note;
  ndichu marked preprint.
- §2: CORTEX/AgentSOC/SIABENCH marked preprint where load-bearing.
- §cost: $25/$29 safety-factor derivation moved to Appendix A (dedup).
