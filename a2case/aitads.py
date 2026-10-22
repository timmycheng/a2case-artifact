"""AIT-ADS external-validity dataset (Zenodo 8263181) -- harness integration.

Extraction: the Suricata alerts in AIT-ADS v1 ship as verbatim EVE alert
records embedded in each scenario's ``<scenario>_wazuh.json`` under ``data``
(provenance ``location: /var/log/suricata/eve.json``). We filter those
records and map them into the same Alert schema the CIC emulation and
ingest.load_eve produce: family = the ruleset's own classification category,
signature_id = the real ET Open SID, text carries the entities.

Reference partition (construction rule, memo §2.2 / sign-off ①): *incident =
one stage window with >=1 alert* -- the CIC protocol's base rule (incident =
documented window) applied to the ADS's own stage labels. The ADS's
time_label semantics and its evaluation lineage (Landauer et al.) are
stage-level; the CIC Infiltration exception (merging a campaign's phases)
does not apply because each ADS stage is a *distinct label*, unlike
Infiltration's three phases under one label. Pairing for statistics is by
*scenario stream* (8 instances), not by incident, mirroring CIC where
instances are day streams. Stage-level ARI is informative where execution-
level ARI degenerates (a single-incident stream saturates at 0 for every
window-bounded arm). Scenarios whose attack alerts concentrate in one stage
(shaw) are structurally degenerate and reported as such (the CIC replay
precedent). Ground truth is time-attribution only (labels.csv has no
participants); endpoint filtering is sensitivity-check material, not the
main rule. BLOCK_GAP_SEC is kept for stream-structure reporting only.
"""

import csv
import json
import os
import zipfile
from datetime import datetime

from .signatures import Alert

ZIP_PATH = os.environ.get(
    "AITADS_ZIP",
    os.path.join(os.path.expanduser("~"), "aitads_gate0", "ait_ads.zip"))
LABELS_PATH = os.environ.get(
    "AITADS_LABELS",
    os.path.join(os.path.expanduser("~"), "aitads_gate0", "labels.csv"))
STREAM_DIR = os.environ.get(
    "AITADS_STREAMS",
    os.path.join(os.path.expanduser("~"), "aitads_gate0", "streams"))

SCENARIOS = ("fox", "harrison", "russellmitchell", "santos",
             "shaw", "wardbeck", "wheeler", "wilson")
BLOCK_GAP_SEC = 3600.0


def parse_eve_ts(s):
    for pat in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, pat).timestamp()
        except ValueError:
            continue
    return None


def load_windows(labels_path=LABELS_PATH):
    """{scenario: [ {attack,start,end,block} ]}, blocks chained by gap."""
    by_scen = {}
    for r in csv.DictReader(open(labels_path, encoding="utf-8")):
        by_scen.setdefault(r["scenario"], []).append(
            {"attack": r["attack"], "start": float(r["start"]),
             "end": float(r["end"])})
    for scen in by_scen:
        ws = sorted(by_scen[scen], key=lambda w: w["start"])
        b = 0
        for i, w in enumerate(ws):
            w["block"] = b
            if i and w["start"] - ws[i - 1]["end"] > BLOCK_GAP_SEC:
                b += 1
                w["block"] = b
    return by_scen


def incident_of(scen, w):
    return f"{scen}.{w['attack']}"


def _build_alerts(zf, member, windows):
    alerts = []
    n_native = n_skipped = 0
    for line in zf.open(member):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            n_skipped += 1
            continue
        data = rec.get("data") or {}
        al = data.get("alert")
        if not (al and data.get("event_type") == "alert"):
            n_native += 1
            continue
        ts = parse_eve_ts(data.get("timestamp", ""))
        if ts is None:
            n_skipped += 1
            continue
        src = str(data.get("src_ip", ""))
        dst = str(data.get("dest_ip", ""))
        try:
            port = int(data.get("dest_port") or 0)
        except (TypeError, ValueError):
            port = 0
        proto = str(data.get("proto", "TCP"))
        sig = str(al.get("signature", ""))
        fam = str(al.get("category", "")).strip().upper() or "UNKNOWN"
        try:
            sev = int(al.get("severity") or 3)
        except (TypeError, ValueError):
            sev = 3
        sid = f"ET-{al.get('signature_id', 0)}"
        hit = None
        for w in windows:
            if w["start"] <= ts <= w["end"]:
                hit = w
                break
        alerts.append(Alert(
            alert_id=0, ts=ts, src=src, dst=dst, dst_port=port, proto=proto,
            sid=sid, family=fam, text=f"{sig} [{proto} {src} -> {dst}:{port}]",
            severity=sev, is_attack=hit is not None,
            gt_incident=incident_of(member.split("_")[0], hit)
            if hit else ""))
    alerts.sort(key=lambda a: a.ts)
    for i, a in enumerate(alerts):
        a.alert_id = i + 1
    return alerts, n_native, n_skipped


def build_scenario(zip_path=ZIP_PATH, scen="harrison",
                   windows_by_scen=None):
    """(alerts, incidents, stats) for one scenario, straight from the zip."""
    windows_by_scen = windows_by_scen or load_windows()
    windows = windows_by_scen[scen]
    zf = zipfile.ZipFile(zip_path)
    alerts, n_native, n_skipped = _build_alerts(
        zf, f"{scen}_wazuh.json", windows)
    incidents = sorted({a.gt_incident for a in alerts if a.is_attack})
    by_block = {}
    for a in alerts:
        if a.is_attack:
            by_block[a.gt_incident] = by_block.get(a.gt_incident, 0) + 1
    stats = dict(scenario=scen, alerts_total=len(alerts),
                 attack_alerts=sum(a.is_attack for a in alerts),
                 fp_alerts=sum(not a.is_attack for a in alerts),
                 wazuh_native=n_native, skipped=n_skipped,
                 incidents=len(incidents),
                 per_incident={k: by_block[k] for k in sorted(by_block)})
    return alerts, incidents, stats


def export_streams(zip_path=ZIP_PATH, stream_dir=STREAM_DIR,
                   windows_by_scen=None):
    """One pass over the zip -> per-scenario alert JSONL (fast reloads)."""
    os.makedirs(stream_dir, exist_ok=True)
    windows_by_scen = windows_by_scen or load_windows()
    zf = zipfile.ZipFile(zip_path)
    summary = {}
    for scen in SCENARIOS:
        out = os.path.join(stream_dir, f"{scen}.jsonl")
        alerts, n_native, n_skipped = _build_alerts(
            zf, f"{scen}_wazuh.json", windows_by_scen[scen])
        with open(out, "w", encoding="utf-8") as f:
            for a in alerts:
                f.write(json.dumps(a.to_dict()) + "\n")
        summary[scen] = dict(n=len(alerts),
                             attack=sum(a.is_attack for a in alerts),
                             incidents=len({a.gt_incident for a in alerts
                                            if a.is_attack}),
                             wazuh_native=n_native, skipped=n_skipped)
        print(f"{scen:16s} exported {len(alerts):>7} alerts "
              f"({summary[scen]['attack']:>6} attack, "
              f"{summary[scen]['incidents']} incidents)", flush=True)
    with open(os.path.join(stream_dir, "manifest.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def load_stream(scen, stream_dir=STREAM_DIR):
    """Fast path: read an exported stream back into Alerts."""
    alerts = []
    with open(os.path.join(stream_dir, f"{scen}.jsonl"), encoding="utf-8") as f:
        for line in f:
            alerts.append(Alert(**json.loads(line)))
    incidents = sorted({a.gt_incident for a in alerts if a.is_attack})
    return alerts, incidents
