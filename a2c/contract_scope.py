"""Direction B (journal plan, Omar reply section 4, 2026-09-27): does the
S3 sharp optimum established on the D2 twin stress test generalize across
scenarios, or is it D2-specific knowledge?

The six-variant summary ablation (S1/S2/S3/S3P/S4/S4M) has only ever been
run on D2 (the concurrency stress test). If S3 were optimal only there,
the "information contract is the binding constraint" claim would be
overstrong -- scenario-specific knowledge. This study runs the same six
contracts over the six non-D2 scenarios (A1/A2/A3/B2/B3/E) -- 36 runs,
zero pipeline changes -- and reports where the contract matters.

Design invariants (fixed before running):
  * scenarios: the paper's six non-D2 originals, rebuilt bit-identically
    via scenarios.build_scenario (seed 17), the same builders as every
    paper row;
  * conditions: S1 (deployed), S2, S3, S4 via classics.stream_summary;
    S3P/S4M via the s4_reversion builders (byte-identical blocks);
  * model: Qwen2.5-0.5B-Instruct, greedy, max_new_tokens 96, same
    prompt/validation/fallback path as s4_reversion;
  * deployment: fp16 on GPU. The scale ablation established this is
    bit-identical to the paper's fp32-CPU deployment for this model on
    all seven original scenarios; the S1 condition re-anchors against
    the round-3 CPU rows in-batch (ARI must match to 1e-12, else the
    scenario is flagged for a CPU re-run);
  * expected per the plan: non-D2 scenarios have no concurrency
    pressure, so the likely outcome is a 1.000 ceiling for every
    condition -- reported as the benchmark's discriminative range: D2 is
    the necessary stress test, and the contract claim is scoped to
    concurrent instances.

Entry point:
    python -m a2c.contract_scope --data <csv-dir> \
        --out results/contract_scope.json [--device cuda]
"""

import argparse
import json

from . import metrics, scenarios
from .arms import _windowize
from .classics import run_hybrid_precomputed, stream_summary
from .locals import LocalLLMPolicyGen
from .round3 import DATA, QWEN
from .s4_reversion import _block_for_gen, s3plus_summary, s4_matrix_text

CONDITIONS = [
    ("S1", lambda w: stream_summary(w, level=1)),
    ("S2", lambda w: stream_summary(w, level=2)),
    ("S3", lambda w: stream_summary(w, level=3)),
    ("S4", lambda w: stream_summary(w, level=4)),
    ("S3P", s3plus_summary),
    ("S4M", s4_matrix_text),
]

SCENARIOS = ["A1", "A2", "A3", "B2", "B3", "E"]


def run(zf, out_path, device="cuda", dtype_name="fp16"):
    import torch

    from .locals import _LLM_USER
    rows = []
    g = LocalLLMPolicyGen(QWEN, device=device, dtype=torch.float16)
    for name in SCENARIOS:
        sc = scenarios.build_scenario(zf, name, seed=17)
        windows = _windowize(sc.alerts, 300)
        n = len(sc.alerts)
        for cond, builder in CONDITIONS:
            pre = (g.prompt_tokens, g.completion_tokens, g.calls, g.fallbacks)
            pols = []
            for w in windows:
                user = _LLM_USER.format(
                    summary=_block_for_gen(builder, w, "local"))
                pols.append(g.generate_policy_text(user, n_window=len(w)))
            res = run_hybrid_precomputed(sc.alerts, pols)
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            rows.append(dict(
                part="contract_scope", scenario=name,
                arm=f"qwen0.5B_{cond}", condition=cond, seed=17,
                deployment=f"{dtype_name}-{device}", n_alerts=n,
                n_incidents=len(sc.incidents), n_windows=len(windows),
                **ev, llm_valid=(g.calls - pre[2]) - (g.fallbacks - pre[3]),
                llm_calls=g.calls - pre[2], fallbacks=g.fallbacks - pre[3],
                prompt_tokens=g.prompt_tokens - pre[0],
                completion_tokens=g.completion_tokens - pre[1],
                dims_hist=json.dumps(sorted({tuple(t["dims"])
                                             for t in res["policy_trace"]}))))
            json.dump(rows, open(out_path, "w"), indent=1)
            print(f"  [{name} {cond}] ari={ev['ari']:.4f} "
                  f"fb={g.fallbacks - pre[3]}/{g.calls - pre[2]}", flush=True)
    print(f"contract_scope complete: {len(rows)} rows", flush=True)


def analyze(out_path="results/contract_scope.json",
            round3_path="results/round3_local.json"):
    import numpy as np
    rows = json.load(open(out_path))
    new = {(r["scenario"], r["condition"]): r["ari"] for r in rows}
    ref_map = {}
    for r in json.load(open(round3_path)):
        if r["part"] == "original_local" and r["arm"] == "hybrid_llm_0.5B":
            ref_map[r["scenario"]] = r["ari"]

    conds = [c for c, _ in CONDITIONS]
    print(f"{'scen':5s} " + " ".join(f"{c:>7s}" for c in conds) + "   S1-anchor")
    mism = []
    for s in SCENARIOS:
        cells = " ".join(f"{new[(s, c)]:7.4f}" for c in conds)
        a1, r3 = new[(s, "S1")], ref_map.get(s)
        ok = (r3 is not None and abs(a1 - r3) < 1e-12)
        if not ok:
            mism.append(s)
        print(f"{s:5s} {cells}   "
              f"{'ok' if ok else 'DIFF(%.6f vs %.6f)' % (a1, r3 or -1)}")
    print("\nS1 anchor vs round-3 CPU rows:",
          "ALL BIT-IDENTICAL" if not mism else f"MISMATCH {mism}")
    for c in conds:
        v = np.array([new[(s, c)] for s in SCENARIOS])
        print(f"  {c:4s} mean={v.mean():.4f} min={v.min():.4f} "
              f"ceil={int((v > 0.9999).sum())}/6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default="results/contract_scope.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        run(args.data, args.out, device=args.device)
    analyze(args.out)


if __name__ == "__main__":
    main()
