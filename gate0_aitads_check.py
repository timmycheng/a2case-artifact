"""Gate 0 verification of AIT-ADS (Zenodo 8263181) -- streaming, in-memory.

Suricata alerts live inside the per-scenario Wazuh JSONL (each record's
``data`` field holds the verbatim Suricata EVE alert record; provenance
``location == /var/log/suricata/eve.json``). This script streams the zip,
extracts the EVE layer, and runs the four checks of revision-plan §2:

  1. format  -- EVE field presence vs the harness contract (a2i.ingest.load_eve)
  2. labels  -- labels.csv attack-window attribution: coverage, ambiguity
  3. scale   -- paired instances under the paper's rule (one incident per
                documented attack execution) vs CIC's 24
  4. noise   -- normal-behavior (true FP) alerts per scenario and per day

Usage: python gate0_aitads_check.py <ait_ads.zip> <labels.csv> <out_json>
"""
import collections
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

EPOCH = datetime(1970, 1, 1)
SCENARIOS = ("fox", "harrison", "russellmitchell", "santos",
             "shaw", "wardbeck", "wheeler", "wilson")


def parse_ts(s):
    for pat in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, pat).timestamp()
        except ValueError:
            continue
    return None


def load_windows(labels_csv):
    by_scen = {}
    import csv
    for r in csv.DictReader(open(labels_csv, encoding="utf-8")):
        by_scen.setdefault(r["scenario"], []).append(
            {"attack": r["attack"], "start": float(r["start"]),
             "end": float(r["end"])})
    for s in by_scen:
        by_scen[s].sort(key=lambda w: w["start"])
    return by_scen


def stream_suricata(zf, member):
    """Yield (ts_epoch, data_dict) for records carrying a verbatim EVE alert.

    Also counts the non-Suricata Wazuh alerts so the multi-IDS picture is
    quantified for the memo."""
    n_wazuh_native = 0
    for line in zf.open(member):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        data = rec.get("data") or {}
        al = data.get("alert")
        if al and data.get("event_type") == "alert":
            ts = parse_ts(data.get("timestamp", "")) or parse_ts(
                rec.get("@timestamp", ""))
            yield ts, data, al, False
        else:
            n_wazuh_native += 1
    # generator must report the native count; done via final yield sentinel
    yield None, None, None, n_wazuh_native


def attribute(ts, windows, offset):
    t = ts + offset
    for w in windows:
        if w["start"] <= t <= w["end"]:
            return w
    return None


def main(zip_path, labels_csv, out_json):
    wins_by = load_windows(labels_csv)
    report = {"scenarios": {}, "format": {}, "scale": {}, "noise": {}}

    eve_fields = collections.Counter()
    sig_counter = collections.Counter()
    cat_counter = collections.Counter()

    for scen in SCENARIOS:
        member = f"{scen}_wazuh.json"
        wins = [dict(w, hits=0) for w in wins_by[scen]]
        ts_all = []
        n_suri = 0
        n_native_final = 0
        stage_hits = collections.Counter()
        for ts, data, al, native_final in stream_suricata(
                zipfile.ZipFile(zip_path), member):
            if ts is None:  # sentinel
                n_native_final = native_final
                break
            n_suri += 1
            ts_all.append(ts)
            if n_suri <= 5000:  # field census on a prefix (cheap, stable)
                for f in ("timestamp", "src_ip", "src_port", "dest_ip",
                          "dest_port", "proto", "app_proto", "flow_id"):
                    if f in data:
                        eve_fields[f] += 1
                for f in ("signature", "signature_id", "category",
                          "severity", "action"):
                    if f in (al or {}):
                        eve_fields["alert." + f] += 1
                sig_counter[al.get("signature", "?")] += 1
                cat_counter[al.get("category", "?")] += 1
            w = attribute(ts, wins, 0.0)
            if w:
                stage_hits[w["attack"]] += 1
                w["hits"] = stage_hits[w["attack"]]
        ts_all.sort()
        in_win = sum(stage_hits.values())
        n_fp = n_suri - in_win
        span_days = (ts_all[-1] - ts_all[0]) / 86400 if ts_all else 0
        # per-day FP rate over the full alert span
        report["scenarios"][scen] = {
            "n_suricata_alerts": n_suri,
            "n_wazuh_native_alerts": n_native_final,
            "first_seen_utc": datetime.fromtimestamp(
                ts_all[0], tz=timezone.utc).isoformat() if ts_all else None,
            "last_seen_utc": datetime.fromtimestamp(
                ts_all[-1], tz=timezone.utc).isoformat() if ts_all else None,
            "span_days": round(span_days, 2),
            "alerts_in_attack_windows": in_win,
            "alerts_normal_behavior_fp": n_fp,
            "fp_per_day": round(n_fp / span_days, 1) if span_days else None,
            "n_stage_windows": len(wins),
            "stage_hits": dict(stage_hits),
            "stages_with_zero_alerts": [
                w["attack"] for w in wins if w["hits"] == 0],
        }
        print(f"{scen:16s} suri={n_suri:>7} native={n_native_final:>7} "
              f"in_win={in_win:>7} fp={n_fp:>7} days={span_days:5.1f} "
              f"zero_stages={report['scenarios'][scen]['stages_with_zero_alerts']}",
              flush=True)

    n = 5000
    report["format"] = {
        "suricata_carrier": ("verbatim EVE alert records embedded in "
                             "<scenario>_wazuh.json under 'data', provenance "
                             "/var/log/suricata/eve.json"),
        "field_presence_in_first_%d" % n: dict(eve_fields),
        "missing_vs_harness_contract": [
            f for f in ("timestamp", "src_ip", "dest_ip", "alert.signature",
                        "alert.signature_id")
            if eve_fields.get(f, 0) < 0.95 * n],
        "top_signatures": sig_counter.most_common(12),
        "top_categories": cat_counter.most_common(8),
    }
    n_exec = len(SCENARIOS)
    report["scale"] = {
        "attack_executions": n_exec,
        "stage_windows_total": sum(
            len(v) for v in wins_by.values()),
        "paired_instances_paper_rule": n_exec,
        "cic_paired_instances": 24,
        "power_note": ("exact two-sided Wilcoxon at n=8: all-8 same sign "
                       "p=0.0078; 7/8 p~0.078; 6/8 p~0.29"),
    }
    json.dump(report, open(out_json, "w", encoding="utf-8"), indent=1,
              ensure_ascii=False)
    print("wrote", out_json)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
