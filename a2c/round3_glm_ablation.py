"""Run ONLY the GLM summary ablation (S1-S4 on D2), appending to round3_glm.json."""
import json, traceback
from a2c import scenarios, metrics, arms
from a2c.classics import GLMPolicyGen, run_hybrid_precomputed
from a2c.round3 import GLM_KEY, DATA

out_path = "results/round3_glm.json"
rows = json.load(open(out_path))
done = {(r["scenario"], r["arm"]) for r in rows if r["part"] == "summary_ablation"}
zf = DATA
sc = scenarios.build_twins(zf, seed=17)
from a2c.arms import _windowize
windows = _windowize(sc.alerts, 300)
n = len(sc.alerts)
try:
    for level in (3, 4):
        arm = f"glm_summary_S{level}"
        if (sc.name, arm) in done or ("D2", arm) in done:
            print(f"skip {arm}")
            continue
        g = GLMPolicyGen(GLM_KEY, summary_level=level)
        pols = g.policies_parallel(windows, workers=2)
        res = run_hybrid_precomputed(sc.alerts, pols)
        res["prompt_tokens"] = g.prompt_tokens
        res["completion_tokens"] = g.completion_tokens
        ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
        c = metrics.cost_row(dict(res, policy_calls=len(pols),
                                  wall=res.get("wall", 0.0)), n)
        dims_hist = json.dumps(sorted(set(tuple(t["dims"]) for t in res["policy_trace"])))
        rows.append(dict(part="summary_ablation", scenario="D2", arm=arm,
                         seed=17, level=level, n_alerts=n,
                         n_incidents=len(sc.incidents), **ev,
                         usd_per_10k=c["usd_per_10k"],
                         llm_valid=g.calls - g.fallbacks, llm_calls=g.calls,
                         dims_hist=dims_hist))
        print(f"[S{level} glm] ari={ev['ari']:.3f} dims={dims_hist} "
              f"valid={g.calls-g.fallbacks}/{g.calls}", flush=True)
        json.dump(rows, open(out_path, "w"), indent=1)
finally:
    json.dump(rows, open(out_path, "w"), indent=1)
print("glm ablation complete")
