import os
"""Round-2 experiments: static dim sweep, D2 twins, LLM-in-the-loop grid,
MiniLM embedding variant, E multi-seed variance. Writes round2.json."""

import json
import sys

from . import scenarios, arms, metrics
from .locals import LocalLLMPolicyGen, MiniLMEmbedder

QWEN = os.environ.get("A2CASE_QWEN_DIR", "../data/models/Qwen2.5-0.5B-Instruct")
MINILM = os.environ.get("A2CASE_MINILM_DIR", "../data/models/all-MiniLM-L6-v2")

DIM_SETS = [
    ("src,dst", ["src", "dst"]),
    ("src,dst,port", ["src", "dst", "dst_port"]),
    ("dst,port", ["dst", "dst_port"]),
    ("family", ["family"]),
    ("family,dst", ["family", "dst"]),
    ("sid", ["sid"]),
    ("src,family", ["src", "family"]),
]
SCEN_NAMES = ["A1", "A2", "A3", "B2", "B3", "E"]
DATA = os.environ.get("A2CASE_DATA_DIR") or os.path.expanduser(
    os.path.join("~", ".cache", "kagglehub", "datasets", "chethuhn",
    "network-intrusion-dataset", "versions", "1"))


def main(out_path):
    zf = DATA
    rows = []

    # --- 1. static dimension-set sweep (incl. D2) ----------------------
    for name in SCEN_NAMES + ["D2"]:
        sc = (scenarios.build_twins(zf, seed=17) if name == "D2"
              else scenarios.build_scenario(zf, name, seed=17))
        n = len(sc.alerts)
        for label, dims in DIM_SETS:
            res = arms.run_static(sc.alerts, dims=dims)
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            rows.append(dict(part="static_sweep", scenario=name,
                             config=label, seed=17, n_alerts=n,
                             n_incidents=len(sc.incidents), **ev))
        print("static sweep done:", name, flush=True)

    # --- 2. D2 twins across all three original arms --------------------
    sc = scenarios.build_twins(zf, seed=17)
    n = len(sc.alerts)
    for arm, res in [("embedding_w2v", arms.run_embedding(sc.alerts)),
                     ("hybrid_psh", arms.run_hybrid(sc.alerts))]:
        ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
        cost = metrics.cost_row(res, n)
        rows.append(dict(part="d2_arms", scenario="D2", arm=arm, seed=17,
                         n_alerts=n, n_incidents=len(sc.incidents),
                         **ev, **cost))
    print("D2 arms done", flush=True)

    # --- 3. LLM-in-the-loop grid (real local model) --------------------
    gen = LocalLLMPolicyGen(QWEN)
    for name in SCEN_NAMES + ["D2"]:
        sc = (scenarios.build_twins(zf, seed=17) if name == "D2"
              else scenarios.build_scenario(zf, name, seed=17))
        n = len(sc.alerts)
        g = LocalLLMPolicyGen(QWEN)   # fresh counters per scenario
        res = arms.run_hybrid(sc.alerts, generator=g)
        ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
        cost = metrics.cost_row(dict(res, prompt_tokens=g.prompt_tokens,
                                     completion_tokens=g.completion_tokens,
                                     policy_calls=g.calls), n)
        rows.append(dict(part="llm_grid", scenario=name, arm="hybrid_llm_0.5B",
                         seed=17, n_alerts=n,
                         n_incidents=len(sc.incidents), **ev, **cost,
                         llm_valid=g.calls - g.fallbacks, llm_calls=g.calls,
                         llm_wall=round(g.wall, 1),
                         dims_hist=json.dumps(sorted(set(
                             tuple(t["dims"]) for t in g.trace)))))
        print(f"LLM grid done: {name} valid {g.calls-g.fallbacks}/{g.calls} "
              f"ari={ev['ari']:.3f}", flush=True)

    # --- 4. LLM multi-seed on E (variance) ------------------------------
    for seed in (17, 29, 43, 61, 79):
        sc = scenarios.build_scenario(zf, "E", seed=seed)
        n = len(sc.alerts)
        g = LocalLLMPolicyGen(QWEN)
        res = arms.run_hybrid(sc.alerts, generator=g)
        ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
        rows.append(dict(part="llm_e_seeds", scenario="E",
                         arm="hybrid_llm_0.5B", seed=seed, n_alerts=n,
                         n_incidents=len(sc.incidents), **ev,
                         llm_valid=g.calls - g.fallbacks,
                         llm_calls=g.calls))
        # reference arms for the same seeds
        for arm, res2 in [("static", arms.run_static(sc.alerts)),
                          ("hybrid_psh", arms.run_hybrid(sc.alerts))]:
            ev2 = metrics.evaluate(res2["cases"], n, len(sc.incidents))
            rows.append(dict(part="llm_e_seeds", scenario="E", arm=arm,
                             seed=seed, n_alerts=n,
                             n_incidents=len(sc.incidents), **ev2))
        print(f"E seeds done: seed {seed}", flush=True)

    # --- 5. MiniLM embedding variant ------------------------------------
    enc = MiniLMEmbedder(MINILM)
    for name in ["A1", "A2", "A3", "B2", "B3", "E", "D2"]:
        sc = (scenarios.build_twins(zf, seed=17) if name == "D2"
              else scenarios.build_scenario(zf, name, seed=17))
        n = len(sc.alerts)
        best = None
        for tau in (0.3, 0.4, 0.5, 0.6, 0.7):
            res = arms.run_embedding(sc.alerts, tau=tau, encoder=enc)
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            if best is None or ev["ari"] > best[1]["ari"]:
                best = (tau, ev)
        tau, ev = best
        rows.append(dict(part="minilm", scenario=name,
                         arm="embedding_MiniLM_best_tau", seed=17, tau=tau,
                         n_alerts=n, n_incidents=len(sc.incidents), **ev))
        print(f"MiniLM done: {name} tau={tau} ari={ev['ari']:.3f}", flush=True)

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    print("round2 complete:", len(rows), "rows ->", out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/round2.json")
