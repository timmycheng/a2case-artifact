"""AIT-ADS full matrix: three arms + deployment-baseline keys x 8 scenarios.

Protocol transfer from the CIC study (revision-plan §3.5): identical arms,
identical metrics (metrics.evaluate), identical statistics conventions
(Wilcoxon signed-rank on instance-level ARI differences, zero differences
dropped, exact null when no ties). Instances = scenario streams; shaw is
structurally degenerate for ARI (single attack stage) and reported as such,
mirroring the CIC replay precedent.

Run:  python -m a2i.aitads_matrix [out_json]
"""
import json
import sys
import time

from . import aitads, arms, metrics
from .static_sweep import DIM_SETS


def main(out_path="results/gate0/aitads_matrix.json"):
    rows = []
    per_scen_policy_mix = {}
    for scen in aitads.SCENARIOS:
        alerts, incidents = aitads.load_stream(scen)
        n, n_inc = len(alerts), len(incidents)
        n_attack = sum(a.is_attack for a in alerts)
        print(f"== {scen}: {n} alerts ({n_attack} attack), "
              f"{n_inc} stage-incidents", flush=True)

        def row(arm, dimset, res):
            ev = metrics.evaluate(res["cases"], n, n_inc)
            r = dict(scenario=scen, arm=arm, dimset=dimset, n_alerts=n,
                     n_attack=n_attack, n_incidents=n_inc,
                     wall=round(res["wall"], 3),
                     policy_calls=res.get("policy_calls", 0),
                     prompt_tokens=res.get("prompt_tokens", 0),
                     completion_tokens=res.get("completion_tokens", 0),
                     fallbacks=res.get("fallbacks", 0), **ev)
            rows.append(r)
            print(f"   [{arm:>9s} {dimset:>13s}] ari={ev['ari']:.3f} "
                  f"purity={ev['purity']:.3f} over={ev['over']:.2f} "
                  f"under={ev['under']:.2f}", flush=True)
            return r

        for label, dims in DIM_SETS:
            row("static", label, arms.run_static(alerts, dims=dims))

        row("embedding", "w2v-tau.75", arms.run_embedding(alerts))

        t0 = time.perf_counter()
        res = arms.run_hybrid(alerts, generator=None)
        r = row("hybrid-PSH", "per-window", res)
        trace = res["policy_trace"]
        per_scen_policy_mix[scen] = dict(
            windows=len(trace),
            dims_mix={d: sum(1 for t in trace if ",".join(t["dims"]) == d)
                      for d in sorted({",".join(t["dims"]) for t in trace})},
            window_mix={str(w): sum(1 for t in trace
                                    if t["window_min"] == w)
                        for w in sorted({t["window_min"] for t in trace})},
            wall=round(time.perf_counter() - t0, 1))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(rows=rows, policy_mix=per_scen_policy_mix), f, indent=1)
    print("wrote", out_path)
    return rows


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else \
        "results/gate0/aitads_matrix.json"
    main(out)
