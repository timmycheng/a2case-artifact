# Window parameter sensitivity on the sustained Thursday campaign (direction C) — results & writing notes

**Executed 2026-09-28.** Data: `results/window_sweep.json`;
runner: `a2i/window_sweep.py` (oracle candidate-timeout parameterization
added to `classics.OracleGen`, default unchanged).

## 1. Burst anatomy (executed, verification-controlled)

R-thursday's single true incident (23,767 attack alerts, all on one
(src,dst) C2 channel, 4 signatures: ET-2023207 × 23,754 plus three
sparse malware sids × 13) decomposes as:

| segment | n alerts | span | first (UTC) |
| --- | --- | --- | --- |
| precursor | 2 | instant | 17:19:45 |
| main burst | 23,762 | 11.7 min (intra-burst gaps < 30 s) | 17:28:51 |
| trailer | 3 | instant | 18:05:10 |

Quiet channels: precursor→burst 9.1 min; burst→trailer 24.6 min.
Total campaign span 45.4 min.

## 2. Sweep result (t = case inactivity timeout)

| t (min) | incumbent ARI | incumbent burst cases | oracle@t ARI | oracle@t burst cases |
| --- | --- | --- | --- | --- |
| 1 | 0.000 | 3 | 0.000 | 3 |
| 5 | 0.000 | 3 | 0.000 | 3 |
| 10 | 0.000 | 2 | 0.000 | 2 |
| 15 | 0.000 | 2 | 0.000 | 2 |
| 20 | 0.000 | 2 | 0.000 | 2 |
| 30 | **1.000** | 1 | **1.000** | 1 |

- Recovery threshold within the swept grid: **t = 30 min**; implied
  minimum by the largest quiet channel: **25 min**.
- The oracle restricted to timeout t equals the incumbent at every t:
  dimension choice cannot compensate for window semantics —
  fragmentation is orthogonal to the grouping key.
- Case-count arithmetic matches the anatomy exactly (t<9.1: 3 cases;
  9.1≤t<24.6: 2; t≥24.6: 1).

## 3. Interpretation (as written into results §replay-results)

- A practical recovery threshold exists in the swept range and is ~3×
  the deployment default (10 min): sustained C2 campaigns need
  inactivity windows at campaign scale. The executor contract accepts
  timeouts up to 120 min (the LLM policy schema allows 1–120), but no
  candidate space in the study offers >10.
- Where no acceptable threshold exists, Julisch-style generalization
  (no time window) is the correct tool for that regime.
- Footing disclosed: single-incident stream, ARI coarse there; the
  value is the quantified mechanism, not arm ranking.

## 4. Figure

`paper/figures/window_sweep.pdf` (step plot, both arms + 24.6-min
dotted line), generated inline 2026-09-28.
