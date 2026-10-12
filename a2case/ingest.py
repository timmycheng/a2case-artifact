"""Ingest CIC-IDS2017 flow CSVs -> alert stream + case-level ground truth."""

import random
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd

from . import schedule
from .signatures import (FP_SIGNATURES, P_DETECT, SIGNATURES, Alert,
                         _sid_for, attack_alert_for, fp_alert_for)

EPOCH = datetime(1970, 1, 1)


def _ts(dt):
    return (dt - EPOCH).total_seconds()


_CANON = {}
import re as _re

def _canon(s):
    return _re.sub(r"[^A-Za-z0-9]+", " ", str(s)).strip().lower()

for _k, _v in schedule.LABEL_ALIASES.items():
    _CANON.setdefault(_canon(_k), _v)


def normalize_label(raw):
    """Robust label normalization: the en-dash in 'Web Attack – XSS' arrives
    mojibake'd depending on the CSV's encoding; canonicalize on
    alphanumeric skeleton so every variant maps to the schedule label."""
    return _CANON.get(_canon(str(raw)), str(raw).strip())


def load_day(zf, member: str) -> pd.DataFrame:
    """Read one day CSV. ``zf`` is either an opened ZipFile or a base
    directory string (the released harness accepts both layouts)."""
    import os
    if isinstance(zf, str):
        df = pd.read_csv(os.path.join(zf, member), low_memory=False,
                         encoding="utf-8")
    else:
        df = pd.read_csv(zf.open(member), low_memory=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Label": "Label"})
    df["Label"] = df["Label"].map(normalize_label)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Flow Duration", "Total Fwd Packets"])
    return df


def _incident_id(label: str, ts: float) -> str:
    """Map (label, timestamp) to a ground-truth incident id.

    Each documented attack window is one incident. Infiltration is one
    label: the dataset treats the whole Thursday campaign (metasploit
    exploit, cool-disk, Dropbox fetch) as one attack instance, so its three
    documented phases map to a single incident --- grouping them is the
    desired analyst behavior.
    """
    if label == "Infiltration":
        return "infiltration"
    return label.replace(" ", "-").lower()


def build_stream(zf, member: str, day_key: str, fp_per_attack: float,
                 seed: int = 17, max_rows: int | None = None,
                 cap: int = 1500, jitter: float = 0.0,
                 twin: dict | None = None):
    """Return (alerts sorted by ts, ground-truth incident set, stats).

    fp_per_attack: expected benign false-positive alerts per attack alert
    (the noise ratio parameter). cap: per-incident alert cap (Suricata
    thresholding semantics). jitter: symmetric relative expansion of each
    documented window (ground truth stays label-consistent). twin: optional
    dict(label=..., src=..., frac=...) injecting a concurrent same-family
    campaign from a second attacker (distinct gt incident '<label>-twin').
    """
    df = load_day(zf, member)
    if max_rows:
        df = df.head(max_rows)
    rng = random.Random(seed)
    hosts = schedule.TOPOLOGY

    day_attacks = {lab: meta for lab, meta in schedule.ATTACKS.items()
                   if meta["day"] == day_key}
    attack_rows = df[df["Label"].isin(day_attacks)]
    benign_rows = df[df["Label"] == "BENIGN"]

    alerts = []
    aid = 0

    # --- attack alerts -------------------------------------------------
    # per-incident alert cap: production IDS apply thresholding/rate filters
    # (Suricata threshold semantics); we emulate with a uniform seeded
    # subsample so a single volumetric incident cannot dominate the stream.
    INCIDENT_CAP = cap
    for label, meta in day_attacks.items():
        sub = attack_rows[attack_rows["Label"] == label]
        s, e = meta["window"]
        s_ts, e_ts = _ts(s), _ts(e)
        if jitter > 0:
            pad = (e_ts - s_ts) * jitter
            s_ts, e_ts = s_ts - pad, e_ts + pad
        p = P_DETECT[label]
        n_fire = int(min(INCIDENT_CAP, max(1, round(len(sub) * p))))
        if len(sub) == 0:
            continue
        idx = sorted(rng.sample(range(len(sub)), n_fire))
        twin_here = twin is not None and twin.get("label") == label
        for i in idx:
            row = sub.iloc[i]
            # reconstruct timestamp inside the documented window; heavier
            # tails for scan-style campaigns
            frac = rng.random() ** (0.6 if label in ("PortScan", "DDoS") else 1.0)
            ts = s_ts + frac * (e_ts - s_ts)
            port = int(row.get("Destination Port", meta["port"]) or meta["port"])
            proto = "UDP" if label == "DDoS" and rng.random() < 0.3 else "TCP"
            src = rng.choice(meta["src"])
            aid += 1
            hint = f"flow_dur={int(row['Flow Duration'])} fwd_pkts={int(row['Total Fwd Packets'])}"
            alerts.append(attack_alert_for(
                label, ts, rng, port, proto, src, meta["dst"],
                _incident_id(label, ts), aid, hint))
            if twin_here and rng.random() < twin.get("frac", 0.5):
                aid += 1
                alerts.append(attack_alert_for(
                    label, ts + rng.uniform(0.2, 2.0), rng, port, proto,
                    twin["src"], meta["dst"], label.replace(" ", "-").lower()
                    + "-twin", aid, hint))

    n_attack_alerts = len(alerts)

    # --- false positives on benign traffic ------------------------------
    n_fp = int(round(fp_per_attack * n_attack_alerts))
    day_start, day_end = _ts(datetime(2017, 7, 4 if day_key == "tuesday" else
                                      5 if day_key == "wednesday" else
                                      6 if day_key.startswith("thursday") else 7,
                                      8, 0)), _ts(datetime(2017, 7, 4 if day_key == "tuesday" else
                                                           5 if day_key == "wednesday" else
                                                           6 if day_key.startswith("thursday") else 7,
                                                           17, 0))
    for _ in range(n_fp):
        ts = rng.uniform(day_start, day_end)
        aid += 1
        alerts.append(fp_alert_for(ts, rng, aid))

    alerts.sort(key=lambda a: a.ts)
    incidents = sorted({a.gt_incident for a in alerts if a.is_attack})
    stats = dict(day=day_key, flows_total=len(df), flows_attack=len(attack_rows),
                 flows_benign=len(benign_rows), attack_alerts=n_attack_alerts,
                 fp_alerts=n_fp, alerts_total=len(alerts),
                 incidents=len(incidents))
    return alerts, incidents, stats


# --- Suricata EVE JSON drop-in path ----------------------------------------
# Real Suricata emits one alert record per line:
#   {"timestamp": "2017-07-04T08:30:12.123456+0000", "event_type": "alert",
#    "src_ip": ..., "src_port": ..., "dest_ip": ..., "dest_port": ...,
#    "proto": "TCP",
#    "alert": {"action": "alerted", "gid": 1, "signature_id": 2210046,
#              "rev": 1, "signature": "ET ... ...", "category": "...",
#              "severity": 2}, ...}
# The mapping below is the contract the PCAP-replay study would use:
# signature text -> (family, severity) via the same catalog the emulation
# layer samples from, so both ingest paths land in one Alert schema. One
# artifact-specific convenience key ("a2i_alert_id") survives the round trip
# for ground-truth re-attachment; loaders outside this project ignore it.

import json as _json

_TEMPLATE_TO_META = {}
for _label, _variants in SIGNATURES.items():
    for _tmpl, _fam, _sev in _variants:
        _TEMPLATE_TO_META[_tmpl] = (_fam, _sev, True)
for _tmpl, _fam, _sev in FP_SIGNATURES:
    _TEMPLATE_TO_META[_tmpl] = (_fam, _sev, False)

_SID_NUM = {t: int(s.split("-")[1]) for t, s in
            ((t, _sid_for(t)) for t in _TEMPLATE_TO_META)}


def alerts_to_eve(alerts, path: str) -> int:
    """Serialize alerts in Suricata EVE alert-record schema (JSON lines).
    Returns the number of records written."""
    from datetime import datetime, timezone
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for a in sorted(alerts, key=lambda x: x.ts):
            dt = datetime.fromtimestamp(a.ts, tz=timezone.utc)
            rec = {
                "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%S.%f+0000"),
                "event_type": "alert",
                "src_ip": a.src,
                "src_port": 0,
                "dest_ip": a.dst,
                "dest_port": a.dst_port,
                "proto": a.proto,
                "alert": {
                    "action": "alerted",
                    "gid": 1,
                    "signature_id": _SID_NUM[a.text.split(" [")[0]],
                    "rev": 1,
                    "signature": a.text.split(" [")[0],
                    "category": a.family,
                    "severity": a.severity,
                },
                "a2i_alert_id": a.alert_id,
            }
            f.write(_json.dumps(rec) + "\n")
            n += 1
    return n


def load_eve(path: str):
    """Ingest real Suricata EVE JSON (alert records, one per line) into the
    same Alert schema the emulation layer produces: signature text maps to
    (family, severity) through the shared catalog; signature_id maps to the
    stable sid; attack vs. benign is derived from the signature's family
    (attack catalogs vs. POLICY/INFO false-positive families). Real
    (non-catalog) signatures keep the ruleset's own classification category
    as the family. Returns
    (alerts sorted by ts, stats) --- ground truth is not part of an EVE file
    and must be re-attached downstream (the round-trip check in
    a2case.eve_check does this via the artifact's alert-id extension key)."""
    n, n_bad, alerts = 0, 0, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                n_bad += 1
                continue
            if rec.get("event_type") != "alert":
                n_bad += 1
                continue
            al = rec.get("alert", {})
            sig = str(al.get("signature", ""))
            meta = _TEMPLATE_TO_META.get(sig)
            if meta is None:
                # real-ET signature (PCAP-replay path): the family field
                # carries the ruleset's own classification category -- the
                # production vocabulary an analyst would key on -- instead
                # of the emulation catalog's family labels. Catalog-matched
                # records (the round-trip path) keep their catalog family.
                fam = (str(al.get("category", "")).strip().upper()
                       or "UNKNOWN")
                sev, atk = int(al.get("severity", 3)), True
            else:
                fam, sev, atk = meta
            ts = datetime.strptime(rec["timestamp"],
                                   "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
            aid = rec.get("a2i_alert_id", n + 1)
            src, dst = rec.get("src_ip", ""), rec.get("dest_ip", "")
            port = int(rec.get("dest_port", 0) or 0)
            proto = str(rec.get("proto", "TCP"))
            sid = f"ET-{al.get('signature_id', _SID_NUM.get(sig, 0))}"
            text = f"{sig} [{proto} {src} -> {dst}:{port}]"
            alerts.append(Alert(alert_id=aid, ts=ts, src=src, dst=dst,
                                dst_port=port, proto=proto, sid=sid,
                                family=fam, text=text, severity=sev,
                                is_attack=atk, gt_incident=""))
            n += 1
    alerts.sort(key=lambda a: a.ts)
    stats = dict(records=n, skipped=n_bad, attacks=sum(a.is_attack for a in alerts),
                 benign=sum(not a.is_attack for a in alerts))
    return alerts, stats
