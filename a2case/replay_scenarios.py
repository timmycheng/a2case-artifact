"""Stage-1 replay scenarios: real Suricata EVE -> pipeline streams.

Replays the CIC-IDS2017 Tue/Wed/Thu PCAPs through Suricata 7.0.3 with the
ET Open ruleset (suricata-update default; 52,768 enabled rules; checksum
validation disabled because the captures carry NIC-offload checksums) and
feeds the resulting EVE alert records into the pipeline through
ingest.load_eve, producing the real-vs-emulated comparison of journal
plan stage 1.

Time base (executed alignment check below): the documented attack windows
in schedule.ATTACKS are wall-clock times in the dataset's local civil time
(Atlantic Daylight Time, UTC-3, July 2017); the PCAP packet timestamps are
UTC. Suricata renders EVE timestamps in the host timezone (+0800 on this
machine), which strptime(%z) normalizes back to correct epochs. The
alignment check reproduces the calibration: the only real attack burst
(SID 2023207, the Infiltration meterpreter C2 channel) falls inside the
Infiltration window converted to UTC.

Ground-truth re-attachment (the replay analogue of the CSV path's
label-derived assignment): an EVE alert is assigned to documented incident
L iff its timestamp falls inside L's window (UTC) AND both endpoints match
L's documented participants (attacker-side set, victim) in either
direction. All other alerts are background alerts carrying no mapping
claim (is_attack=False). This is the same epistemic status as the
emulation's label-derived assignment: dataset labels define ground truth;
the audit machinery of stage 2 quantifies the mapping noise.

Entry point:
    python -m a2case.replay_scenarios --eve <dir-with-eve_*.json> \
        --out results/replay_compare.json [--llm]
"""

import argparse
import json
import os
from datetime import timedelta, timezone

from . import ingest, metrics, scenarios, schedule
from .arms import _windowize
from .ingest import load_eve
from .locals import LocalLLMPolicyGen, MiniLMEmbedder
from .round3 import MINILM, QWEN, free_arms

# The dataset's documented windows are local civil time; July in New
# Brunswick = Atlantic Daylight Time, UTC-3.
ADT = timezone(timedelta(hours=-3))

DEFAULT_EVE_DIR = os.environ.get("A2CASE_EVE_DIR", "./replay")

DAY_FILES = {
    "tuesday": ("eve_tuesday.json",
                [("Tuesday-WorkingHours.pcap_ISCX.csv", "tuesday")]),
    "wednesday": ("eve_wednesday.json",
                  [("Wednesday-workingHours.pcap_ISCX.csv", "wednesday")]),
    "thursday": ("eve_thursday.json",
                 [("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                   "thursday_am"),
                  ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                   "thursday_pm")]),
}


def window_utc(label):
    """Documented window for an attack label, as UTC epoch bounds."""
    s, e = schedule.ATTACKS[label]["window"]
    return (s.replace(tzinfo=ADT).timestamp(),
            e.replace(tzinfo=ADT).timestamp())


def day_labels(day_members):
    """Attack labels documented on the scenario day's members."""
    out = []
    for _member, dkey in day_members:
        out += [lab for lab, meta in schedule.ATTACKS.items()
                if meta["day"] == dkey]
    return out


def attach_ground_truth(alerts, labels):
    """Topology+window assignment (see module docstring). Returns
    per-incident alert counts."""
    inc_counts = {lab: 0 for lab in labels}
    for a in alerts:
        a.gt_incident = ""
        a.is_attack = False
        for lab in labels:
            meta = schedule.ATTACKS[lab]
            ws, we = window_utc(lab)
            if not (ws <= a.ts <= we):
                continue
            if (a.src in meta["src"] and a.dst == meta["dst"]) or \
               (a.dst in meta["src"] and a.src == meta["dst"]):
                a.gt_incident = ingest._incident_id(lab, a.ts)
                a.is_attack = True
                inc_counts[lab] += 1
                break
    return inc_counts


def verify_alignment(alerts):
    """Executed calibration: the dominant attack-side signature burst must
    sit inside its documented window under the ADT->UTC conversion.
    Returns the check record for the methods note."""
    # the burst = most common (sid) among alerts touching documented
    # attacker IPs in either direction
    from collections import Counter
    att = set(schedule.TOPOLOGY["attacker_kali"]) | {
        schedule.TOPOLOGY["attacker_win_69"], schedule.TOPOLOGY["attacker_win_70"],
        schedule.TOPOLOGY["attacker_win_71"]}
    cand = [a for a in alerts if a.src in att or a.dst in att]
    if not cand:
        return dict(found=False)
    sid = Counter(a.sid for a in cand).most_common(1)[0][0]
    burst = [a for a in cand if a.sid == sid]
    ts0, ts1 = burst[0].ts, burst[-1].ts
    # which documented window contains the burst?
    hit = None
    for lab in schedule.ATTACKS:
        ws, we = window_utc(lab)
        # allow the burst to start inside the window (span may exceed it)
        if ws <= ts0 <= we or ws <= ts1 <= we:
            hit = lab
            break
    return dict(found=True, sid=sid, n=len(burst),
                first_utc=ts0, last_utc=ts1, window=hit)


def build_replay_scenario(day, eve_dir):
    """Scenario from real EVE for one day: load, attach ground truth,
    report coverage stats alongside the emulated baseline's incidents."""
    fname, members = DAY_FILES[day]
    alerts, stats = load_eve(os.path.join(eve_dir, fname))
    labels = day_labels(members)
    inc_counts = attach_ground_truth(alerts, labels)
    align = verify_alignment(alerts)
    incidents = sorted({a.gt_incident for a in alerts if a.is_attack})
    stats.update(scenario=f"R-{day}", day=day, records=stats.get("records", 0),
                 attack_alerts=sum(inc_counts.values()),
                 incidents=len(incidents),
                 incidents_emulated=len(labels),
                 per_incident={lab: inc_counts[lab] for lab in labels},
                 alignment=align)
    return scenarios.Scenario(f"R-{day}", alerts, incidents, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eve", default=DEFAULT_EVE_DIR)
    ap.add_argument("--out", default="results/replay_compare.json")
    ap.add_argument("--llm", action="store_true",
                    help="also run the 0.5B LLM arm (local, greedy)")
    args = ap.parse_args()
    rows = []
    minilm = MiniLMEmbedder(MINILM)
    for day in ("tuesday", "wednesday", "thursday"):
        sc = build_replay_scenario(day, args.eve)
        print(f"== {sc.name}: records={sc.stats['records']} "
              f"incidents={sc.stats['incidents']}"
              f"/{sc.stats['incidents_emulated']} per-incident="
              f"{sc.stats['per_incident']}", flush=True)
        d = free_arms(sc, minilm)
        if args.llm:
            g = LocalLLMPolicyGen(QWEN)
            d["hybrid_llm_0.5B"] = arms_run_hybrid(sc, g)
        degenerate = []
        for arm, res in d.items():
            if res is None:
                # julisch hindsight grid: with zero attack alerts every
                # candidate scores NaN and no best exists -- the day is
                # degenerate for arm comparison by construction
                degenerate.append(arm)
                continue
            n = len(sc.alerts)
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            rows.append(dict(
                part="replay", scenario=sc.name, arm=arm, day=day,
                n_alerts=n, n_incidents=len(sc.incidents),
                incidents_emulated=sc.stats["incidents_emulated"],
                attack_alerts=sc.stats["attack_alerts"],
                per_incident=json.dumps(sc.stats["per_incident"]),
                alignment=json.dumps(sc.stats["alignment"]),
                degenerate=arm in degenerate, **ev))
        if degenerate:
            print(f"  [degenerate, skipped] {degenerate}", flush=True)
        json.dump(rows, open(args.out, "w"), indent=1)


def arms_run_hybrid(sc, g):
    from . import arms
    res = arms.run_hybrid(sc.alerts, generator=g)
    res["prompt_tokens"] = g.prompt_tokens
    res["completion_tokens"] = g.completion_tokens
    return res


if __name__ == "__main__":
    main()
