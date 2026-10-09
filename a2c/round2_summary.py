"""Summarize round2.json for the paper."""

import json
from collections import defaultdict


def main(path="results/round2.json"):
    rows = json.load(open(path))
    print("=== static sweep winners (incl D2) ===")
    by_scen = defaultdict(list)
    for r in rows:
        if r["part"] == "static_sweep":
            by_scen[r["scenario"]].append(r)
    for s, rs in by_scen.items():
        best = max(rs, key=lambda r: r["ari"])
        worst = min(rs, key=lambda r: r["ari"])
        print(f"  {s}: best={best['config']}({best['ari']:.2f}) "
              f"worst={worst['config']}({worst['ari']:.2f})")

    print("\n=== LLM grid ===")
    for r in rows:
        if r["part"] == "llm_grid":
            print(f"  {r['scenario']}: ari={r['ari']:.3f} over={r['over']:.2f} "
                  f"under={r['under']:.2f} valid={r['llm_valid']}/{r['llm_calls']} "
                  f"usd10k={r['usd_per_10k']:.4f} wall={r['llm_wall']}s "
                  f"dims={r['dims_hist']}")

    print("\n=== E multi-seed (ARI by arm) ===")
    seeds = defaultdict(dict)
    for r in rows:
        if r["part"] == "llm_e_seeds":
            seeds[r["arm"]][r["seed"]] = r["ari"]
    for arm, d in seeds.items():
        vals = [d[s] for s in sorted(d)]
        print(f"  {arm}: {[round(v,3) for v in vals]} "
              f"mean={sum(vals)/len(vals):.3f}")

    print("\n=== MiniLM vs W2V (ARI) ===")
    res1 = json.load(open("results/results.json"))["results"]
    for r in rows:
        if r["part"] == "minilm":
            w2v = [x for x in res1 if x["scenario"] == r["scenario"]
                   and x["arm"] == "embedding" and x["seed"] == 17]
            w2v_ari = w2v[0]["ari"] if w2v else float("nan")
            print(f"  {r['scenario']}: MiniLM(tau={r['tau']})={r['ari']:.3f} "
                  f"vs W2V={w2v_ari:.3f}")

    print("\n=== D2 arms ===")
    for r in rows:
        if r["part"] == "d2_arms":
            print(f"  {r['arm']}: ari={r['ari']:.3f} over={r['over']:.2f} "
                  f"under={r['under']:.2f}")


if __name__ == "__main__":
    main()
