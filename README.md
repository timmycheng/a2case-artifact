# A2Case — Alerts-to-Cases Artifact

Artifact for the paper **"Case is all you need: A measurement study of
alert-to-case grouping policies on public IDS data"** (Omar Dib and
Cheng Bao).

A2Case is a deterministic alert-to-case grouping engine: alerts are normalized,
sliced into micro-batch windows, and grouped into cases under a grouping
policy that is **selected per window** from an auditable candidate space.
This repository contains everything needed to regenerate every table and
figure in the paper from one command per artifact: the grouping pipeline,
the session-to-incident ground-truth mapping, the scenario generators, the
pre-registration documents, and all raw per-run results.

## Repository layout

```
a2case/          grouping pipeline and experiment harnesses (Python package)
results/      all raw per-run results (JSON) and analysis logs
docs/         design notes: replay, S4 mechanism, scale ablation, contract
              gap/scope, incumbent sensitivity, window sensitivity
prereg/       pre-registration documents (frozen before execution)
verify_tab_main.py   re-checks Table 7 against results/results.json
verify_tab_llm.py    re-checks Table 8 against results/*.json
```

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (torch, transformers, numpy, scipy,
  pandas, scikit-learn, matplotlib; CPU torch is sufficient — the 0.5B
  anchor ran on CPU)
- Optional: [Suricata 7.0.3](https://suricata.io) with the ET Open ruleset
  (via `suricata-update`) for the real-capture replay
  (`docs/REPLAY_DESIGN.md`)

## Data acquisition

1. **CIC-IDS2017 flow CSVs** (emulation + benchmark streams): public
   distribution of the Canadian Institute for Cybersecurity. The harness
   reads the dataset directory pointed to by `A2CASE_DATA_DIR` (default: the
   standard kagglehub cache of `chethuhn/network-intrusion-dataset`,
   version 1).
2. **Source PCAPs** (real-capture replay): the five working days
   (52.4 GB) were obtained on 2026-09-21 from a public mirror of the
   dataset (Hugging Face repository `bvsam/cic-ids-2017`, via
   hf-mirror.com) because the official distribution requires registration.
   Per-day MD5 checksums published by the mirror were verified on download;
   byte-equality with the original release was not verifiable without
   registration, so the packet-level findings in the paper are statements
   about this acquisition chain (documented in `docs/REPLAY_DESIGN.md`).
3. **Suricata replay**: ET Open ruleset at `suricata-update` defaults
   (snapshot September 2026, 52,768 enabled rules), EVE alert records as
   the pipeline input. See `docs/REPLAY_DESIGN.md` for the full replay
   configuration (checksum validation off for NIC-offloaded captures; UTC
   vs. Atlantic-time reconciliation).

## Environment variables

All machine-specific locations are configurable; nothing is hardcoded.

| Variable | Purpose |
|---|---|
| `A2CASE_DATA_DIR` | CIC-IDS2017 dataset directory (CSV per day) |
| `GLM_API_KEY` | Zhipu/Z.ai API key, needed only for the GLM-5.3 frontier condition |
| `A2CASE_LLM_BASE_URL` / `A2CASE_LLM_API_KEY` / `A2CASE_LLM_MODEL` | generic OpenAI-compatible endpoint for LLM arms |
| `A2CASE_QWEN_DIR` / `A2CASE_MINILM_DIR` | local model directories (Qwen2.5-0.5B-Instruct; all-MiniLM-L6-v2) |
| `A2CASE_MODEL_ROOT` | root directory holding the Qwen2.5 family for the scale ablation |
| `A2CASE_EVE_DIR` | directory of Suricata EVE JSON records for the replay scenarios |

## Reproducing the paper

Run from the repository root (`python -m a2case.<module>`); each module writes
its outputs into `results/`. The mapping below covers the paper's main
artifacts (each module's docstring has the exact flags):

| Module | Output | Paper artifact |
|---|---|---|
| `a2case.run` | `results/results.json` | main grid (Table 7) |
| `a2case.static_sweep` | `results/static_sweep.json` | fixed-configuration sweep (§6, Fig 3) |
| `a2case.round3` / `a2case.round3_resume` | `results/round3_local.json` | three-generator round incl. D2 stress test |
| `a2case.round3_glm_ablation` | `results/round3_glm.json` | frontier condition (Table 8) |
| `a2case.mixed_expansion` | `results/round3_expansion.json` | pre-registered mixed-suite expansion (P25–P36) |
| `a2case.round3_stats` | subgroup/interaction analyses | Tables 12–13, §7.9 |
| `a2case.s4_reversion` | `results/s4_reversion_*.json` | summary-contract mechanism study (Table 10) |
| `a2case.contract_scope` | `results/contract_scope.json` | contract scope across scenarios (Table 11) |
| `a2case.contract_gap_n20` | `results/contract_gap_n20.json` | pre-registered n=20 contract contrast (§7.8) |
| `a2case.incumbent_sensitivity` | `results/incumbent_sensitivity.json` | comparator sensitivity (Table 14) |
| `a2case.replay_scenarios` | `results/replay_compare*.json` | real-Suricata replay (Tables 15–16) |
| `a2case.window_sweep` | `results/window_sweep.json` | inactivity-timeout sweep (Fig 4) |
| `a2case.scale_ablation` | `results/scale_ablation.json` | same-family scale curve (Table 9) |
| `a2case.case_cards` | `results/case_cards.json` | worked case cards (Appendix C) |
| `a2case.audit_sample` / `a2case.audit_sensitivity` | `results/audit*/` | ground-truth audit + sensitivity (§5) |
| `a2case.eve_check` | `results/eve_roundtrip_*.jsonl` | EVE ingest round-trip check |
| `a2case.figures` | figures | paper figures (PDF) |

Verification (tables vs. stored results):

```
python verify_tab_main.py
python verify_tab_llm.py
```

The pre-registration documents in `prereg/` were frozen before the
corresponding batches were executed; `docs/` records design decisions and
claim scoping for each study thread.

## Determinism and seeds

Published seeds are 17, 29, 43 (seed 17 is reported in the tables). Static
arms are seed-exact by construction; the 0.5B LLM arm is greedy and
deterministic given weights and prompt; the GLM-5.3 frontier condition
offers no snapshot pinning, so its run-to-run variation is disclosed in the
paper (Tables 8 and 10 captions) and all request/response traces ship with
the results.

## License

MIT — see [LICENSE](LICENSE).
