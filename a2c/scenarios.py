"""Scenario construction (advisor-scoped: A, B, E only)."""

import zipfile
from dataclasses import dataclass

from . import schedule, ingest

# scenario instances: (name, day_key, zip member, noise ratio)
SCENARIO_INSTANCES = [
    ("A1", "tuesday", "Tuesday-WorkingHours.pcap_ISCX.csv", 1.0),
    ("A2", "wednesday", "Wednesday-workingHours.pcap_ISCX.csv", 1.0),
    ("A3", "friday", "FRIDAY_MULTI", 1.0),          # Bot + PortScan + DDoS
    ("B1", "tuesday", "Tuesday-WorkingHours.pcap_ISCX.csv", 8.0),
    ("B2", "wednesday", "Wednesday-workingHours.pcap_ISCX.csv", 8.0),
    ("B3", "friday", "FRIDAY_MULTI", 8.0),
    ("E", "thursday", "THURSDAY_MULTI", 1.0),       # multi-stage kill chain
]

FRIDAY_MEMBERS = ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]
THURSDAY_MEMBERS = ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"]

DAY_KEYS = {"tuesday": "tuesday", "wednesday": "wednesday",
            "Friday-WorkingHours-Morning.pcap_ISCX.csv": "friday_am",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "friday_ps",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "friday_ddos",
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "thursday_am",
            "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "thursday_pm"}


@dataclass
class Scenario:
    name: str
    alerts: list
    incidents: list
    stats: dict


def build_twins(zf, seed: int = 17, twin_alerts: int = 600):
    """Concurrent-twins stress test (controlled synthetic construction,
    the pilot's analogue of the proposal's Scenario D).

    Base: the Wednesday stream (Scenario A2). A second, *concurrent* DoS
    Hulk campaign from a different external attacker (205.174.165.70) is
    interleaved into the original Hulk window against the same victim.
    Ground truth: the twin is a distinct incident. This creates the regime
    the benchmark cannot express --- two incidents sharing one signature
    family, one victim, and one time window --- under which every
    family-keyed fixed rule over-merges. The construction is synthetic and
    labeled as such; main claims rest on the untransformed scenarios.
    """
    from .signatures import SIGNATURES, _sid_for, Alert
    import random

    base = build_scenario(zf, "A2", seed=seed)
    base_alerts, incidents, stats = base.alerts, base.incidents, base.stats
    rng = random.Random(seed + 1)
    hulk = [a for a in base_alerts
            if a.gt_incident == "dos-hulk"]
    src_sample = [a.src for a in hulk]
    twin = []
    aid = max(a.alert_id for a in base_alerts)
    tmpl, fam, sev = SIGNATURES["DoS Hulk"][0]
    # draw twin alert timestamps interleaved in the same window
    for a in hulk[:twin_alerts]:
        aid += 1
        twin.append(Alert(
            alert_id=aid, ts=a.ts + rng.uniform(0.5, 2.5),
            src="205.174.165.70", dst=a.dst, dst_port=a.dst_port,
            proto="TCP", sid=_sid_for(tmpl), family=fam, severity=sev,
            text=f"{tmpl} [TCP 205.174.165.70 -> {a.dst}:{a.dst_port}] "
                 f"flow_dur={a.text.split('flow_dur=')[1]}",
            is_attack=True, gt_incident="dos-hulk-twin"))
    alerts = sorted(base_alerts + twin, key=lambda a: a.ts)
    incidents = sorted(set(incidents) | {"dos-hulk-twin"})
    stats = dict(stats)
    stats.update(scenario="D2", noise_ratio=stats.get("noise_ratio", 1.0),
                 alerts_total=len(alerts), incidents=len(incidents),
                 twin_alerts=len(twin), construction="synthetic-twin")
    return Scenario("D2", alerts, incidents, stats)


def _merge(streams):
    alerts = [a for s in streams for a in s[0]]
    alerts.sort(key=lambda a: a.ts)
    incidents = sorted({i for s in streams for i in s[1]})
    n_att = sum(s[2]["attack_alerts"] for s in streams)
    n_fp = sum(s[2]["fp_alerts"] for s in streams)
    stats = dict(day="+".join(s[2]["day"] for s in streams),
                 attack_alerts=n_att, fp_alerts=n_fp,
                 alerts_total=len(alerts), incidents=len(incidents))
    return alerts, incidents, stats


def build_scenario(zf: zipfile.ZipFile, name: str, seed: int = 17) -> Scenario:
    inst = next(s for s in SCENARIO_INSTANCES if s[0] == name)
    _, day_key, member, ratio = inst
    if member == "FRIDAY_MULTI":
        streams = [ingest.build_stream(zf, m, DAY_KEYS[m], ratio, seed)
                   for m in FRIDAY_MEMBERS]
        alerts, incidents, stats = _merge(streams)
    elif member == "THURSDAY_MULTI":
        streams = [ingest.build_stream(zf, m, DAY_KEYS[m], ratio, seed)
                   for m in THURSDAY_MEMBERS]
        alerts, incidents, stats = _merge(streams)
    else:
        alerts, incidents, stats = ingest.build_stream(zf, member, day_key,
                                                       ratio, seed)
    stats["scenario"] = name
    stats["noise_ratio"] = ratio
    return Scenario(name, alerts, incidents, stats)
