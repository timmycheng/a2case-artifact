"""Static dimension-set sweep: does ANY fixed rule configuration win everywhere?

This is the experiment that answers the reviewer question "why not hard-code
the best static rule?": every fixed dimension set that looks good in one
regime fails in another. Run per scenario x dim-set, seed 17.
"""

import json
import sys

from . import scenarios, arms, metrics

DIM_SETS = [
    ("src,dst", ["src", "dst"]),
    ("src,dst,port", ["src", "dst", "dst_port"]),
    ("dst,port", ["dst", "dst_port"]),
    ("family", ["family"]),
    ("family,dst", ["family", "dst"]),
    ("sid", ["sid"]),
    ("src,family", ["src", "family"]),
]


def run(zip_or_dir, out_path, seed=17):
    rows = []
    for name in ["A1", "A2", "A3", "B2", "B3", "E"]:
        sc = scenarios.build_scenario(zip_or_dir, name, seed=seed)
        n = len(sc.alerts)
        for label, dims in DIM_SETS:
            res = arms.run_static(sc.alerts, dims=dims)
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            rows.append(dict(scenario=name, dimset=label, seed=seed,
                             n_alerts=n, n_incidents=len(sc.incidents),
                             **ev))
            print(f"{name} {label:14s} ari={ev['ari']:.3f} over={ev['over']:.2f} "
                  f"under={ev['under']:.2f}")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    # per-scenario winner check
    print("\nwinner per scenario:")
    for name in ["A1", "A2", "A3", "B2", "B3", "E"]:
        best = max((r for r in rows if r["scenario"] == name),
                   key=lambda r: r["ari"])
        worst = min((r for r in rows if r["scenario"] == name),
                    key=lambda r: r["ari"])
        print(f"  {name}: best={best['dimset']} ({best['ari']:.2f}), "
              f"worst={worst['dimset']} ({worst['ari']:.2f})")
    return rows


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
