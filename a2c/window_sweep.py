"""Direction C (journal plan, Omar reply section 4, 2026-09-27): window
parameter sensitivity on the Thursday sustained-C2 burst.

Background (stage-1 replay, REPLAY_DESIGN.md section 6): on R-thursday
the only real incident is the Infiltration C2 burst -- 23,7k alerts on
one channel in ~12 minutes. Every windowed policy arm, including the
oracle, fragments it (ARI 0.000) while Julisch generalization holds it
(ARI 1.000). That is currently a behavioral observation with a mechanism
story but no quantification. This study converts it into a design
parameter result: how large an inactivity timeout must the executor's
policy space offer for sustained-campaign bursts to survive?

Design (fixed before running):
  * stream: the full Thursday EVE replay (R-thursday, single day, no new
    data); ground truth attached exactly as in stage 1;
  * burst anatomy first: inter-alert gap structure of the attack alerts
    (the burst is a single incident, so the whole question is the largest
    quiet gap the executor must bridge);
  * sweep: inactivity timeout t in {1, 5, 10, 15, 20, 30} min, two arms
    per t --
      - static incumbent (src,dst) at timeout t: isolates pure
        inactivity semantics (fixed key, no policy choice involved);
      - oracle restricted to timeout t (same DIM_SETS search, greedy
        per-window global-ARI selection, only the timeout candidate set
        is the swept singleton): adds the policy-selection layer on top;
  * metric: ARI on the attack-alert restriction (as everywhere in the
    paper) plus burst_cases (number of distinct cases containing attack
    alerts -- the direct fragmentation statistic);
  * recovery threshold: the smallest swept t at which burst_cases = 1.

Entry point:
    python -m a2c.window_sweep --eve <dir> --out results/window_sweep.json
"""

import argparse
import json
import math

from . import metrics
from .arms import run_static
from .classics import OracleGen
from .replay_scenarios import build_replay_scenario, DEFAULT_EVE_DIR

SWEEP = [1, 5, 10, 15, 20, 30]


def burst_anatomy(sc):
    """Gap structure of the attack alerts (single incident by
    construction on R-thursday)."""
    atk = sorted((a for a in sc.alerts if a.is_attack), key=lambda a: a.ts)
    gaps = [(b.ts - a.ts) for a, b in zip(atk, atk[1:])]
    gaps.sort(reverse=True)
    span = atk[-1].ts - atk[0].ts if len(atk) > 1 else 0.0
    top = gaps[:10]
    return dict(n_attack=len(atk),
                span_min=span / 60.0,
                span_utc_first=atk[0].ts if atk else None,
                span_utc_last=atk[-1].ts if atk else None,
                top_gaps_min=[g / 60.0 for g in top],
                max_gap_min=top[0] / 60.0 if top else 0.0,
                max_gap_seconds=top[0] if top else 0.0,
                implied_min_timeout_min=(math.ceil(top[0] / 60.0)
                                         if top else 0),
                keys=len({(a.src, a.dst) for a in atk}),
                sids=len({a.sid for a in atk}))


def sweep_arm(sc, t):
    """Both arms at timeout t. Returns per-arm record."""
    out = {}
    n = len(sc.alerts)

    def rec(res):
        ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
        idx = {i for i, c in enumerate(res["cases"])
               if any(a.is_attack for a in c)}
        ev["burst_cases"] = len(idx)
        ev["cases_total"] = len(res["cases"])
        return ev

    st = run_static(sc.alerts, dims=["src", "dst"], window_min=t)
    out["incumbent"] = rec(st)
    og = OracleGen(window_min_choices=(t,))
    part = og.run(sc.alerts)
    out["oracle"] = rec(dict(cases=part))
    out["oracle"]["trace"] = og.trace
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eve", default=DEFAULT_EVE_DIR)
    ap.add_argument("--out", default="results/window_sweep.json")
    args = ap.parse_args()

    sc = build_replay_scenario("thursday", args.eve)
    anat = burst_anatomy(sc)
    print("R-thursday built: records=%d incidents=%d attack=%d" % (
        sc.stats["records"], sc.stats["incidents"], anat["n_attack"]),
        flush=True)
    print("burst anatomy: span=%.2f min  max_gap=%.4f min (%.1f s)  "
          "top_gaps(min)=%s" % (
              anat["span_min"], anat["max_gap_min"], anat["max_gap_seconds"],
              [round(g, 4) for g in anat["top_gaps_min"][:5]]), flush=True)

    rows = [dict(part="burst_anatomy", scenario="R-thursday", **anat)]
    print(f"\n{'t(min)':>6s} | {'inc ARI':>8s} {'inc cases':>9s} | "
          f"{'orc ARI':>8s} {'orc cases':>9s}")
    for t in SWEEP:
        r = sweep_arm(sc, t)
        rows.append(dict(part="window_sweep", scenario="R-thursday",
                         timeout_min=t,
                         incumbent=r["incumbent"], oracle=r["oracle"]))
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"{t:6d} | {r['incumbent']['ari']:8.4f} "
              f"{r['incumbent']['burst_cases']:9d} | "
              f"{r['oracle']['ari']:8.4f} {r['oracle']['burst_cases']:9d}",
              flush=True)
    rec = next((r for r in rows if r["part"] == "window_sweep"
                and r["incumbent"]["burst_cases"] == 1), None)
    if rec:
        print(f"\nrecovery threshold (incumbent): t = {rec['timeout_min']} min "
              f"(implied by max gap: {anat['implied_min_timeout_min']} min)",
              flush=True)
    else:
        print("\nno swept timeout recovers a single burst case for the "
              "incumbent arm", flush=True)
    print("window_sweep complete:", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
