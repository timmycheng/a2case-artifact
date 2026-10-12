"""Resume: parametric suite (24 instances) for the local arms only."""

import json
import sys

from a2case import scenarios, arms, metrics
from a2case.classics import OracleGen, run_julisch, DIM_SETS
from a2case.locals import LocalLLMPolicyGen, MiniLMEmbedder
from a2case.round3 import build_parametric, QWEN, MINILM, DATA


def main(out_path="results/round3_local.json"):
    rows = json.load(open(out_path))
    done = {(r["scenario"], r["arm"]) for r in rows if r["part"] == "parametric"}
    minilm = MiniLMEmbedder(MINILM)

    def free_arms(sc, n):
        d = {}
        d["static_incumbent"] = arms.run_static(sc.alerts, dims=["src", "dst"])
        best, best_ari = None, -1
        for label, dims in DIM_SETS:
            r = arms.run_static(sc.alerts, dims=dims)
            ev = metrics.evaluate(r["cases"], n, len(sc.incidents))
            if best_ari < 0 or ev["ari"] > best_ari:
                best, best_ari = r, ev["ari"]
        d["static_best_fixed_hindsight"] = best
        d["julisch"] = run_julisch(sc.alerts)
        jb, jb_ari = None, -1
        for wmin, cap in [(10, 11), (30, 11), (10, 1), (30, 3)]:
            r = run_julisch(sc.alerts, window_min=wmin, raise_cap=cap)
            ev = metrics.evaluate(r["cases"], n, len(sc.incidents))
            if ev["ari"] > jb_ari:
                jb, jb_ari = r, ev["ari"]
        d["julisch_best_fixed_hindsight"] = jb
        d["embedding_w2v"] = arms.run_embedding(sc.alerts)
        d["embedding_minilm"] = arms.run_embedding(sc.alerts, encoder=minilm)
        d["hybrid_psh"] = arms.run_hybrid(sc.alerts)
        og = OracleGen()
        import time
        t0 = time.perf_counter()
        part = og.run(sc.alerts)
        d["oracle"] = dict(cases=part, wall=time.perf_counter() - t0,
                           policy_calls=og.calls, prompt_tokens=0,
                           completion_tokens=0, fallbacks=0, policy_trace=og.trace)
        return d

    for pid in range(1, 25):
        name = f"P{pid:02d}"
        sc = build_parametric(DATA, pid, seed=17)
        n = len(sc.alerts)
        sc.stats["seed"] = 17
        d = free_arms(sc, n)
        if (name, "hybrid_llm_0.5B") not in done:
            g = LocalLLMPolicyGen(QWEN)
            d["hybrid_llm_0.5B"] = arms.run_hybrid(sc.alerts, generator=g)
            d["hybrid_llm_0.5B"]["prompt_tokens"] = g.prompt_tokens
            d["hybrid_llm_0.5B"]["completion_tokens"] = g.completion_tokens
        for arm, res in d.items():
            if (name, arm) in done:
                continue
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            extra = {}
            if res.get("prompt_tokens") or res.get("completion_tokens"):
                c = metrics.cost_row(res, n)
                extra = dict(usd_per_10k=c["usd_per_10k"],
                             prompt_tokens=res.get("prompt_tokens", 0),
                             completion_tokens=res.get("completion_tokens", 0))
            rows.append(dict(part="parametric", scenario=name, arm=arm,
                             seed=17, n_alerts=n,
                             n_incidents=len(sc.incidents),
                             **ev, **extra, stats=sc.stats))
        print(f"{name} done ({n} alerts, {len(sc.incidents)} inc)", flush=True)
        json.dump(rows, open(out_path, "w"), indent=1)
    print("parametric suite complete", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/round3_local.json")
