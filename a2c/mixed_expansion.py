"""Pre-registered mixed-regime expansion: instances P25-P36 (n=8 -> 20).

Implements exactly the generator specification frozen in
prereg_mixed_expansion.md (2026-09-21): three new twin configurations
(C7/C8/C9) drawn from the already-used grid values, crossed with the four
scenario days, under the round-3 twin rule (day's volumetrically largest
family, second attacker 205.174.165.70, frac 0.5, seed 17). Execution is a
single batch over all 12 instances with the identical arm composition of
round3's parametric part; rows land in results/round3_expansion.json with
part="parametric" so the round-3 statistics read them unchanged.

Run once, after the stage-1 replay conclusions:
    python -m a2c.mixed_expansion --data <csv-dir> \
        --out results/round3_expansion.json
"""

import argparse
import json

from . import arms, round3
from .ingest import build_stream
from .locals import LocalLLMPolicyGen, MiniLMEmbedder
from .round3 import DAYS, MINILM, QWEN, TWIN_SRC, evaluate_all

# frozen in prereg_mixed_expansion.md, 2026-09-21
EXPANSION_CONFIGS = [
    ("C7", 1, 1500, 0.05),
    ("C8", 4, 1000, 0.10),
    ("C9", 16, 750, 0.05),
]
DAY_ORDER = ["tuesday", "wednesday", "thursday", "friday"]
EXPANDED_IDS = [f"P{pid:02d}" for pid in range(25, 37)]


def build_expansion(zf, pid, seed=17):
    """P25-P36: pid 25-28 = C7, 29-32 = C8, 33-36 = C9; within each block
    the day runs tue/wed/thu/fri (the pre-registered table order)."""
    idx = pid - 25
    cfg_name, ratio, cap, jitter = EXPANSION_CONFIGS[idx // 4]
    day = DAY_ORDER[idx % 4]
    from . import schedule
    twin_label = {"tuesday": "FTP-Patator", "wednesday": "DoS Hulk",
                  "thursday": "Web Attack - Brute Force",
                  "friday": "PortScan"}[day]
    streams = []
    for member, dkey in DAYS[day]:
        t = None
        if schedule.ATTACKS.get(twin_label, {}).get("day") == dkey:
            t = dict(label=twin_label, src=TWIN_SRC, frac=0.5)
        streams.append(build_stream(zf, member, dkey, ratio, seed,
                                    cap=cap, jitter=jitter, twin=t))
    alerts = [a for s in streams for a in s[0]]
    alerts.sort(key=lambda a: a.ts)
    incidents = sorted({i for s in streams for i in s[1]})
    stats = dict(day=day, config=cfg_name, ratio=ratio, cap=cap,
                 jitter=jitter, twin=True, prereg="2026-09-21",
                 alerts_total=len(alerts), incidents=len(incidents))
    return round3.scenarios.Scenario(f"P{pid:02d}", alerts, incidents, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/round3_expansion.json")
    args = ap.parse_args()
    zf = args.data
    rows = json.load(open(args.out)) if __import__("os").path.exists(
        args.out) else []
    minilm = MiniLMEmbedder(MINILM)
    for pid in range(25, 37):
        sc = build_expansion(zf, pid, seed=17)
        d = round3.free_arms(sc, minilm)
        g = LocalLLMPolicyGen(QWEN)
        d["hybrid_llm_0.5B"] = arms.run_hybrid(sc.alerts, generator=g)
        d["hybrid_llm_0.5B"]["prompt_tokens"] = g.prompt_tokens
        d["hybrid_llm_0.5B"]["completion_tokens"] = g.completion_tokens
        sc.stats["seed"] = 17
        evaluate_all(f"P{pid:02d}", sc, d, rows, "parametric")
        json.dump(rows, open(args.out, "w"), indent=1)
    print("mixed-regime expansion complete:", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
