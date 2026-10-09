# Stage-1 replay: real Suricata EVE vs. alert emulation — design & findings

**Date: 2026-09-21 (execution day).** This note documents the executed
replay pipeline and its results up to the arm-level comparison table; it is
the source for the paper's methods subsection and the replay comparison
table (journal plan stage 1, deliverable "replay 对照表 + 方法学小节修订").

## 1. Acquisition chain

- PCAPs: all five working days of CIC-IDS2017, 52.4 GB total, at
  `<data-root>\PCAPs`. Downloaded 2026-09-21 from the
  HuggingFace mirror `bvsam/cic-ids-2017` via hf-mirror.com (the official
  UNB distribution requires registration); MD5s verified against the
  mirror's per-day manifests. **Provenance caveat:** the manifests are the
  mirror's own; byte-equality with the original UNB release was not
  verifiable without registration. Packet-level findings below are
  therefore statements about this acquisition chain.
- Replayed days: Tuesday, Wednesday, Thursday (the plan's three scenario
  days; Friday/Monday not replayed — the paper's main scenarios are Tue/
  Wed/Thu/Fri, but stage-1 scope per plan is Tue/Wed/Thu).

## 2. Suricata configuration (executed)

- Suricata 7.0.3, WSL2 Ubuntu 24.04, offline `pcap-file` replay.
- Ruleset: ET Open via `suricata-update` defaults, snapshot 2026-09-21,
  52,768 enabled rules, `suricata -T` clean.
- `eve-log` types reduced to `alert` + `stats` (the pipeline consumes only
  alert records; the default all-events config would emit tens of GB of
  non-alert records).
- **Checksum validation disabled** (`pcap-file checksum-checks: no`,
  `stream checksum-validation: no`). Rationale: the CIC captures were
  taken with NIC checksum offload, so nearly every packet carries an
  invalid wire checksum; with validation on, the engine emits ~18
  invalid-checksum alerts per 100 packets (35,702 of 36,409 alerts in a
  200 MB smoke slice were `SURICATA TCPv4 invalid checksum`), which are
  capture artifacts, not network behavior. Disabling checksum validation
  is the standard production posture for offloaded captures. With it off,
  the smoke slice yields 362 alerts.
- Replay throughput ~27 MB/s/policy-core set (Tue 11.55M packets/395 s);
  outputs at `<data-root>\replay\eve_{tuesday,wednesday,thursday}.json`
  (5.7k / 17.9k / 29.6k alert records).

## 3. Time base (executed alignment check)

- `schedule.ATTACKS` windows are wall-clock times in the dataset's local
  civil time — Atlantic Daylight Time (UTC-3) for July 2017 in New
  Brunswick.
- PCAP packet timestamps are UTC; Suricata renders EVE timestamps in the
  host timezone (+0800 on this machine), which `load_eve` normalizes via
  `%z` back to correct epochs.
- Executed check (`replay_scenarios.verify_alignment`): the dominant
  attack-side signature burst on Thursday — SID 2023207, *ET MALWARE
  Windows Microsoft Windows DOS prompt command Error*, 23,754 alerts,
  all 192.168.10.8 (documented victim) -> 205.174.165.73 (documented
  attacker) — spans 17:28:51–17:40:33 UTC, inside the documented
  Infiltration window 14:19–15:45 ADT = 17:19–18:45 UTC. The +0800
  rendering initially looks like an ~11 h clock offset; after timezone
  normalization there is **no clock skew** — the PCAP time base is UTC.

## 4. Ground-truth re-attachment (the replay analogue of label-derived gt)

An EVE alert is assigned to documented incident *L* iff its timestamp
falls inside *L*'s window (UTC) AND both endpoints match *L*'s documented
participants (attacker-side set, victim) in either direction. All other
alerts are background alerts with no mapping claim (`is_attack=False`).
This is the same epistemic status as the emulation path's assignment
(dataset labels define ground truth); the stage-2 audit machinery
quantifies the mapping noise. Implemented in
`a2i/replay_scenarios.attach_ground_truth`.

Real (non-catalog) signatures keep the ruleset's own classification
category as their `family` field (production vocabulary); catalog-matched
records (the §5.2 round-trip path, re-verified PASS after this change:
12,018 records, 0 field mismatches, byte-identical partitions) keep their
catalog family.

## 5. Coverage findings (the replay's headline result)

| day | documented incidents | incidents with >=1 real alert | real attack alerts | emulated stream (paper, ratio 1.0) |
| --- | --- | --- | --- | --- |
| Tuesday | 2 (FTP/SSH-Patator) | **0** | 0 | A1: 6,000 alerts, 2 incidents |
| Wednesday | 5 (4 DoS + Heartbleed) | **0** | 0 | A2: 12,018 alerts, 5 incidents |
| Thursday | 4 (3 web + Infiltration) | **1** (Infiltration, 23,767 alerts) | 23,767 | E: 2,966 alerts, 4 incidents |

Packet-level verification (tcpdump, BPF, stderr visible, control-tested):

- Tuesday full file: **zero** packets from the documented attacker
  205.174.165.73, and **zero** packets from the entire documented
  attacker subnet 205.174.165.0/24 as source. Controls executed and
  passing: the file reads (first packet 19:53:32 PCAP-clock, a CDPv2
  switch announcement — consistent with an 08:53 ADT capture start under
  the UTC time base), and the control filter
  (`src host 192.168.10.50`) matches normally, so the empty attacker
  result is not a read/filter artifact.
- Wednesday full file: **zero** packets from 205.174.165.73 and **zero**
  external -> victim 192.168.10.50:80 packets: the DoS flood traffic is
  absent from the capture, not merely undetected.
- Thursday contains the documented attacker traffic (the C2 burst above).

Interpretation, split by cause:

1. **Capture gap (dataset)**: the Tue/Wed merged PCAPs contain no attack
   traffic from the documented attacker at all. This is consistent with
   the published critiques of CIC-IDS2017 capture/label consistency
   (Engelen et al., *Troubleshooting an Intrusion Detection Dataset*, 2021)
   and is a property of the distributed artifact, not of our ruleset.
2. **Ruleset gap (tooling)**: on Thursday, the only documented incident
   that fires any rule is detected by a malware-behavior signature on the
   C2 channel (SID 2023207), not by scan/DoS/web-attack rules; the
   brute-force, DoS, and web-attack campaigns produce no ET Open alerts
   anywhere in the three days.

Consequence for the emulation layer: its per-family detection
probabilities (0.55–0.90) describe an idealized, tuned-IDS coverage
assumption. A default production-style ET Open deployment on this capture
detects essentially none of the benchmark's attack labels, and for two of
three days the attack traffic is not even present in the PCAP. The
emulation is therefore **not** retroactively validated as representative
of default-toolchain output; the honest framing (both results reported) is:

- emulation = label-grounded, full-coverage alert stream (the benchmark's
  intended grouping problem, at the dataset's attack labels);
- real default replay = the operational lower bound, where the grouping
  problem collapses to background noise plus one volumetric C2 burst.

## 6. Arm-level comparison on the real streams

`results/replay_compare_llm.json` (part="replay"; all arms of
round3.free_arms + 0.5B LLM; 300 s windows; ARI on the attack-alert
restriction, as everywhere in the paper):

- R-tuesday, R-wednesday: structurally degenerate — zero attack alerts,
  ARI undefined (NaN) for every arm. This degeneracy is the finding.
- R-thursday (single true incident, 23,767 attack alerts + background):

| arm | ARI |
| --- | --- |
| julisch / julisch best-fixed | 1.000 |
| embedding_w2v | 1.000 |
| static incumbent (src,dst) | 0.000 |
| static best-fixed hindsight | 0.000 |
| embedding_minilm | 0.000 |
| hybrid_psh | 0.000 |
| oracle | 0.000 |
| hybrid_llm_0.5B | 0.000 |

  With a single true class, ARI is 1.0 iff the arm keeps the burst in one
  case. The policy-based arms all fragment the 12-minute, 23.7k-alert
  burst (window/inactivity semantics), while Julisch's generalization
  clustering holds it together. Caveat for the paper: single-incident
  ARI is a coarse statistic; the table's value is the coverage contrast,
  not arm ranking.

## 7. Executed commands / reproducibility

- `<data-root>\PCAPs\replay_all.sh` (WSL, root):
  suricata -c /etc/suricata/suricata.yaml -r <pcap> -l <dir> per day;
  run logs + alert counts in `replay/run_*.txt`.
- `python -m a2c.eve_check` — re-run after the ingest change: PASS.
- `python -m a2c.replay_scenarios --eve <data-root>\replay
  --out results/replay_compare.json [--llm]`.
- Packet checks: `scripts/wsl_attacker_check.sh` (alert-side census),
  `scripts/wsl_packet_check.sh` + `wsl_tcpdump_control.sh` (packet-side).

## 8. Feeds into (stage-5 writing)

- Methods subsection: replay setup (this note, §2–§4).
- New table: coverage contrast (§5) + Thursday arm table (§6).
- Threats/limitations: mirror provenance caveat (§1); single-replay
  snapshot; ARI degeneracy on empty days reported as such.
- Narrative: strengthens the benchmark-circularity critique — the
  benchmark's attack labels are not observable in a default-toolchain
  deployment on two of three days, so any grouping evaluation on this
  data is necessarily a label-grounded construction (which is exactly
  what the paper's emulation makes explicit).
