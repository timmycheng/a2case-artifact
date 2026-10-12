"""Round-3 experiment suite (top-venue revision).

A. Parametric scenario suite (24 instances over days x noise x density x
   jitter x concurrency), all free arms + oracle + Julisch + LLM-0.5B.
B. Original scenarios + D2 with GLM-5.3 (frontier condition, parallel).
C. Summary information ablation S1-S4 on D2 (GLM + 0.5B).
Writes round3.json + stats summary.
"""
import os

import json
import sys
from collections import Counter

import numpy as np

from . import scenarios, arms, metrics, ingest
from .classics import (OracleGen, run_julisch, GLMPolicyGen,
                       run_hybrid_precomputed, stream_summary, DIM_SETS)
from .locals import LocalLLMPolicyGen, MiniLMEmbedder

QWEN = os.environ.get("A2CASE_QWEN_DIR", "../data/models/Qwen2.5-0.5B-Instruct")
MINILM = os.environ.get("A2CASE_MINILM_DIR", r"..\data\models\all-MiniLM-L6-v2")

GLM_KEY = os.environ.get("GLM_API_KEY", "")  # set GLM_API_KEY to run the GLM conditions
DATA = os.environ.get("A2CASE_DATA_DIR") or os.path.expanduser(
    os.path.join("~", ".cache", "kagglehub", "datasets", "chethuhn",
    "network-intrusion-dataset", "versions", "1"))

DAYS = {
    "tuesday": [("Tuesday-WorkingHours.pcap_ISCX.csv", "tuesday")],
    "wednesday": [("Wednesday-workingHours.pcap_ISCX.csv", "wednesday")],
    "thursday": [("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", "thursday_am"),
                 ("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "thursday_pm")],
    "friday": [("Friday-WorkingHours-Morning.pcap_ISCX.csv", "friday_am"),
               ("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", "friday_ps"),
               ("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", "friday_ddos")],
}

# 24 parametric instances: day x (ratio, cap, jitter, twin)
CONFIGS = [
    (1, 1500, 0.00, False), (1, 1500, 0.15, False),
    (4, 750, 0.00, False), (8, 1500, 0.10, False),
    (8, 750, 0.15, True), (16, 1000, 0.00, True),
]
TWIN_SRC = "205.174.165.70"


def build_parametric(zf, pid, seed=17):
    day = ["tuesday", "wednesday", "thursday", "friday"][(pid - 1) // 6]
    ratio, cap, jitter, twin_on = CONFIGS[(pid - 1) % 6]
    from . import schedule
    # twin targets the day's volumetrically largest attack family
    twin_label = {"tuesday": "FTP-Patator", "wednesday": "DoS Hulk",
                  "thursday": "Web Attack - Brute Force",
                  "friday": "PortScan"}[day]
    streams = []
    for member, dkey in DAYS[day]:
        t = None
        if twin_on and schedule.ATTACKS.get(twin_label, {}).get("day") == dkey:
            t = dict(label=twin_label, src=TWIN_SRC, frac=0.5)
        streams.append(ingest.build_stream(zf, member, dkey, ratio, seed,
                                           cap=cap, jitter=jitter, twin=t))
    alerts = [a for s in streams for a in s[0]]
    alerts.sort(key=lambda a: a.ts)
    incidents = sorted({i for s in streams for i in s[1]})
    stats = dict(day=day, ratio=ratio, cap=cap, jitter=jitter, twin=twin_on,
                 alerts_total=len(alerts), incidents=len(incidents))
    return scenarios.Scenario(f"P{pid:02d}", alerts, incidents, stats)


def evaluate_all(name, sc, arms_dict, rows, part, llm_meta=None):
    n = len(sc.alerts)
    for arm, res in arms_dict.items():
        ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
        extra = {}
        if res.get("prompt_tokens") or res.get("completion_tokens"):
            c = metrics.cost_row(res, n)
            extra = dict(usd_per_10k=c["usd_per_10k"],
                         prompt_tokens=res.get("prompt_tokens", 0),
                         completion_tokens=res.get("completion_tokens", 0))
        row = dict(part=part, scenario=name, arm=arm, seed=sc.stats.get("seed", 17),
                   n_alerts=n, n_incidents=len(sc.incidents),
                   **ev, **extra)
        if llm_meta:
            row.update(llm_meta)
        rows.append(row)
    print(f"  [{name}] " + " ".join(
        f"{a}={metrics.evaluate(r['cases'], n, len(sc.incidents))['ari']:.3f}"
        for a, r in arms_dict.items()), flush=True)


def free_arms(sc, minilm):
    """The arm composition every parametric instance is run under:
    incumbent, hindsight-best fixed, Julisch (+ hindsight grid), both
    embedding arms, PSH hybrid, oracle."""
    n = len(sc.alerts)
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
    d["julisch_best_fixed_hindsight"] = None
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
    t0 = __import__("time").perf_counter()
    part = og.run(sc.alerts)
    d["oracle"] = dict(cases=part, wall=__import__("time").perf_counter() - t0,
                       policy_calls=og.calls, prompt_tokens=0,
                       completion_tokens=0, fallbacks=0, policy_trace=og.trace)
    return d


def main(zf, out_path, glm=True, local=True, parametric=True):
    rows = []
    minilm = MiniLMEmbedder(MINILM)

    # ---- A/B: original scenarios + D2 ----------------------------------
    original = [("A1", lambda: scenarios.build_scenario(zf, "A1", seed=17)),
                ("A2", lambda: scenarios.build_scenario(zf, "A2", seed=17)),
                ("A3", lambda: scenarios.build_scenario(zf, "A3", seed=17)),
                ("B2", lambda: scenarios.build_scenario(zf, "B2", seed=17)),
                ("B3", lambda: scenarios.build_scenario(zf, "B3", seed=17)),
                ("E", lambda: scenarios.build_scenario(zf, "E", seed=17)),
                ("D2", lambda: scenarios.build_twins(zf, seed=17))]

    if local:
        for name, build in original:
            sc = build()
            d = free_arms(sc, minilm)
            g = LocalLLMPolicyGen(QWEN)
            d["hybrid_llm_0.5B"] = arms.run_hybrid(sc.alerts, generator=g)
            d["hybrid_llm_0.5B"]["prompt_tokens"] = g.prompt_tokens
            d["hybrid_llm_0.5B"]["completion_tokens"] = g.completion_tokens
            evaluate_all(name, sc, d, rows, "original_local",
                         llm_meta=dict(llm_valid=g.calls - g.fallbacks,
                                       llm_calls=g.calls))

    if glm:
        import os
        done_glm = set()
        if os.path.exists(out_path):
            try:
                done_glm = {(r["scenario"], r["arm"]) for r in json.load(open(out_path))}
            except Exception:
                done_glm = set()
        for name, build in original:
            if (name, "hybrid_llm_glm5.3") in done_glm:
                print(f"skip {name} (glm already done)", flush=True)
                continue
            sc = build()
            d = free_arms(sc, minilm)
            g = GLMPolicyGen(GLM_KEY)
            windows = arms._windowize(sc.alerts, 300)
            pols = g.policies_parallel(windows, workers=2)
            d["hybrid_llm_glm5.3"] = run_hybrid_precomputed(sc.alerts, pols)
            d["hybrid_llm_glm5.3"]["prompt_tokens"] = g.prompt_tokens
            d["hybrid_llm_glm5.3"]["completion_tokens"] = g.completion_tokens
            evaluate_all(name, sc, d, rows, "original_glm",
                         llm_meta=dict(llm_valid=g.calls - g.fallbacks,
                                       llm_calls=g.calls))
            json.dump(rows, open(out_path, "w"), indent=1)

    # ---- C: summary information ablation on D2 --------------------------
    if glm:
        sc = scenarios.build_twins(zf, seed=17)
        from .arms import _windowize
        windows = _windowize(sc.alerts, 300)
        for level in (1, 2, 3, 4):
            g = GLMPolicyGen(GLM_KEY, summary_level=level)
            pols = g.policies_parallel(windows, workers=2)
            res = run_hybrid_precomputed(sc.alerts, pols)
            res["prompt_tokens"] = g.prompt_tokens
            res["completion_tokens"] = g.completion_tokens
            n = len(sc.alerts)
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            c = metrics.cost_row(res, n)
            rows.append(dict(part="summary_ablation", scenario="D2",
                             arm=f"glm_summary_S{level}", seed=17, level=level,
                             n_alerts=n, n_incidents=len(sc.incidents),
                             **ev, usd_per_10k=c["usd_per_10k"],
                             llm_valid=g.calls - g.fallbacks,
                             llm_calls=g.calls,
                             dims_hist=json.dumps(sorted(set(
                                 tuple(t["dims"]) for t in res["policy_trace"])))))
            print(f"  [S{level} glm] ari={ev['ari']:.3f} "
                  f"dims={Counter(t['dims'] for t in res['policy_trace']).most_common(3)}",
                  flush=True)
            json.dump(rows, open(out_path, "w"), indent=1)
    if local:
        sc = scenarios.build_twins(zf, seed=17)
        from .arms import _windowize
        windows = _windowize(sc.alerts, 300)
        for level in (1, 2, 3, 4):
            g = LocalLLMPolicyGen(QWEN)
            pols = []
            for w in windows:
                pols.append(g.policy(w, stream_summary(w, level=level)))
            res = run_hybrid_precomputed(sc.alerts, pols)
            res["prompt_tokens"] = g.prompt_tokens
            res["completion_tokens"] = g.completion_tokens
            n = len(sc.alerts)
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            rows.append(dict(part="summary_ablation", scenario="D2",
                             arm=f"qwen_summary_S{level}", seed=17, level=level,
                             n_alerts=n, n_incidents=len(sc.incidents),
                             **ev, llm_valid=g.calls - g.fallbacks,
                             llm_calls=g.calls,
                             dims_hist=json.dumps(sorted(set(
                                 tuple(t["dims"]) for t in res["policy_trace"])))))
            print(f"  [S{level} qwen] ari={ev['ari']:.3f}", flush=True)
            json.dump(rows, open(out_path, "w"), indent=1)

    # ---- D: parametric suite (24 instances) -----------------------------
    if parametric:
        for pid in range(1, 25):
            sc = build_parametric(zf, pid, seed=17)
            d = free_arms(sc, minilm)
            if local:
                g = LocalLLMPolicyGen(QWEN)
                d["hybrid_llm_0.5B"] = arms.run_hybrid(sc.alerts, generator=g)
                d["hybrid_llm_0.5B"]["prompt_tokens"] = g.prompt_tokens
                d["hybrid_llm_0.5B"]["completion_tokens"] = g.completion_tokens
            sc.stats["seed"] = 17
            evaluate_all(f"P{pid:02d}", sc, d, rows, "parametric")
            json.dump(rows, open(out_path, "w"), indent=1)

    with open(out_path, "w") as f:
        json.dump(rows, f, indent=1)
    print("round3 complete:", len(rows), "rows", flush=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/round3.json"
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"
    main(DATA, out,
         glm=mode in ("all", "glm"),
         local=mode in ("all", "local"),
         parametric=mode in ("all", "local", "parametric"))
