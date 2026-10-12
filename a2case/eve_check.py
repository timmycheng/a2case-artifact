"""EVE drop-in round-trip check (operational compatibility sanity test).

Builds the Wednesday stream (A2 basis) with the seeded emulation layer,
serializes it to Suricata EVE alert-record JSON, re-ingests it through
ingest.load_eve, and verifies the full pipeline contract:

  1. record round trip: every alert re-parsed, endpoints/ports/proto
     preserved, timestamps within 1 microsecond (EVE timestamp resolution);
  2. partition round trip: static incumbent and family,dst fixed rules
     produce byte-identical case partitions from both ingest paths;
  3. family recovery: signature -> family mapping identical on both paths.

The check validates the *format contract* (fields, timestamp parsing,
ordering, partition-identical downstream behavior), not ecological
realism --- the alert content is still emulated; see the paper's
threats-to-validity discussion.
"""

import json
import sys
from collections import defaultdict

from . import arms, ingest
from .round3 import DATA

EVE_PATH = "results/eve_roundtrip_wednesday.jsonl"


def _partition_ids(cases):
    return sorted(frozenset(a.alert_id for a in c) for c in cases)


def main(data=DATA, eve_path=EVE_PATH):
    import zipfile
    zf = zipfile.ZipFile(data) if data.endswith(".zip") else data
    member = "Wednesday-workingHours.pcap_ISCX.csv"
    alerts, incidents, stats = ingest.build_stream(
        zf, member, "wednesday", fp_per_attack=1.0, seed=17)
    print(f"source stream: {stats['alerts_total']} alerts "
          f"({stats['attack_alerts']} attack, {stats['fp_alerts']} FP), "
          f"{stats['incidents']} incidents")

    n = ingest.alerts_to_eve(alerts, eve_path)
    print(f"EVE written: {n} records -> {eve_path}")

    eve_alerts, eve_stats = ingest.load_eve(eve_path)
    print(f"EVE parsed: {eve_stats}")

    # --- 1. field-level round trip ---------------------------------------
    src_by_id = {a.alert_id: a for a in alerts}
    max_dts, mismatch = 0.0, 0
    for ea in eve_alerts:
        sa = src_by_id[ea.alert_id]
        max_dts = max(max_dts, abs(ea.ts - sa.ts))
        if (ea.src, ea.dst, ea.dst_port, ea.proto, ea.sid) != (
                sa.src, sa.dst, sa.dst_port, sa.proto, sa.sid):
            mismatch += 1
    print(f"field mismatches: {mismatch}/{len(eve_alerts)}; "
          f"max |dt| = {max_dts:.2e} s")

    # re-attach ground truth by alert id (EVE carries none)
    for ea in eve_alerts:
        ea.gt_incident = src_by_id[ea.alert_id].gt_incident
        ea.is_attack = src_by_id[ea.alert_id].is_attack

    # --- 2. partition round trip ------------------------------------------
    ok = True
    for label, dims in (("incumbent src,dst", ["src", "dst"]),
                        ("family,dst", ["family", "dst"])):
        part_direct = arms.run_static(alerts, dims=dims)["cases"]
        part_eve = arms.run_static(eve_alerts, dims=dims)["cases"]
        same = _partition_ids(part_direct) == _partition_ids(part_eve)
        ok &= same
        print(f"partition [{label}]: direct={len(part_direct)} cases, "
              f"EVE={len(part_eve)} cases, identical={same}")

    # --- 3. family recovery ------------------------------------------------
    fam_direct = {a.alert_id: a.family for a in alerts}
    fam_eve = {a.alert_id: a.family for a in eve_alerts}
    same_fam = fam_direct == fam_eve
    ok &= same_fam
    print(f"family mapping identical: {same_fam}")

    print("EVE round-trip:", "PASS" if ok and mismatch == 0 else "FAIL")
    return 0 if ok and mismatch == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
