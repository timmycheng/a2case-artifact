"""Stratified manual-audit sample for the AIT-ADS reference partition.

Sign-off (1) of revision-plan §3.2: per event class, >=5 examples (or >=10%)
for human verification of the time-attribution rule. Classes = the 10 attack
stage types + the normal-behavior background. Deterministic seed 17.

Output: CSV with one row per sampled alert -- the auditor marks
agree / disagree / unsure in the last column.

Run: python scripts/aitads_audit_sample.py [out_csv] [n_per_class]
"""
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "experiments"))
from a2i import aitads  # noqa: E402


def main(out_csv, n_per_class=5):
    rng = random.Random(17)
    windows = aitads.load_windows()
    by_class = {}
    fp_pool = []
    for scen in aitads.SCENARIOS:
        alerts, _ = aitads.load_stream(scen)
        for a in alerts:
            if a.is_attack:
                stage = a.gt_incident.split(".", 1)[1]
                by_class.setdefault(stage, []).append((scen, a))
            else:
                fp_pool.append((scen, a))
    rows = []
    for stage in sorted(by_class):
        pool = by_class[stage]
        # stratify across scenarios: round-robin scenarios then sample
        rng.shuffle(pool)
        for scen, a in pool[:n_per_class]:
            rows.append([scen, stage, a.gt_incident,
                         datetime.fromtimestamp(a.ts, tz=timezone.utc)
                         .isoformat(),
                         f"{a.src} -> {a.dst}:{a.dst_port}", a.proto,
                         a.sid, a.text[:90], ""])
    rng.shuffle(fp_pool)
    for scen, a in fp_pool[:n_per_class]:
        rows.append([scen, "(background FP)", "",
                     datetime.fromtimestamp(a.ts, tz=timezone.utc).isoformat(),
                     f"{a.src} -> {a.dst}:{a.dst_port}", a.proto,
                     a.sid, a.text[:90], ""])
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# Audit: does each attack-row plausibly belong to the "
                    "attributed stage (timestamp inside the stage window + "
                    "signature/endpoints consistent)? Mark agree/disagree/"
                    "unsure. Background rows must NOT belong to any stage."])
        w.writerow(["scenario", "class", "attributed_incident",
                    "timestamp_utc", "endpoints", "proto", "sid",
                    "signature", "verdict"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_csv} "
          f"({len(by_class)} attack classes x {n_per_class} + {n_per_class} "
          f"background)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), "..", "experiments",
                     "results", "gate0", "audit_sample_aitads.csv")
    main(out, int(sys.argv[2]) if len(sys.argv) > 2 else 5)
