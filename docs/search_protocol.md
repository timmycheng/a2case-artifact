# Related-Work Search Protocol

This document records the literature-search protocol behind the "What the
adjacent literature does not contain" paragraph of the paper's Related Work
section, so the claim can be audited.

## Claim tested

No prior work that (a) evaluates LLM-selected grouping policies executed by
a deterministic engine, or (b) measures grouping policies (fixed or
run-time-selected) against case-level truth on public intrusion-detection
data with partition metrics.

## Sources searched

1. The three LLM-for-SOC surveys cited in the paper (Srinivas et al. 2025;
   Habibzadeh et al. 2026; Ndichu et al. 2026): all cited works and their
   dataset tables.
2. Forward-citation lists (Google Scholar, September 2026) of:
   - COLA (Kuang et al., ICSE-SEIP 2024)
   - CORTEX (Wei et al., arXiv:2510.00311)
   - GARNET (Gong et al., AAAI 2026)
3. Forward-citation list of AIT-ADS (Landauer et al., CSET 2024)
   (Google Scholar, October 2026), added when the dataset-landscape
   paragraph was written. Nearest consumer found: AlertBERT
   (arXiv:2602.06534), cited and differentiated in the paper.
4. Titles of the 2024-2026 programs of the major security conferences
   (USENIX Security, IEEE S&P, ACM CCS, NDSS, ACSAC, DSN, RAID), screened
   for alert grouping / correlation / LLM SOC agent topics.

## Exact queries

- "alert grouping" / "alert correlation" / "alert aggregation", each with
  "LLM"
- "LLM" with "SOC" and with "security operations center"
- "alert triage" with "large language model"
- "run-time policy" with grouping; "dynamic aggregation" with alerts
- "case-level" with evaluation and alerts
- AIT-ADS forward citations: titles screened for grouping/aggregation uses

## Result

No qualifying work found for (a) or (b). Nearest neighbors found and
differentiated in the paper: DEEPCASE, COLA, GARNET, AlertBERT, and the
public alert corpora surveyed in the dataset-landscape paragraph (AIT-ADS,
SALAD-SOC, WitFoo Precinct6, Microsoft GUIDE).
