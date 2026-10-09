"""P2-R case cards (Omar brief, P1 "Analyst-facing metric"): for one
reference incident on each of A2, D2, E, and R-Thursday, dump the case(s)
every load-bearing arm emits --- alert count, time span, entities,
signatures, over-merge or split --- so the appendix can show what the ARI
numbers look like as cases.

Arms per card: incumbent (src,dst,w=10), family (A2/D2; the benchmark
winner), best-fixed (E), PSH, LLM 0.5B at C1 (re-run in-batch on GPU;
partitions were not stored by the round-3 grids), oracle, and on
R-Thursday additionally Julisch (the no-window control that holds).

Deterministic arms recompute exactly; the LLM arm re-runs the deployed
generation path (greedy, w anchor asserted implicitly by identity of the
summary blocks, which are byte-identical to the deployed generator's by
construction --- the same guard as contract_gap).

Entry (from experiments/):
    python -m a2c.case_cards [--device cuda]
Output: results/case_cards.json + printed digest.
"""

import argparse
import json
import time
from collections import Counter

from . import arms, classics, metrics, round3, scenarios
from .arms import _windowize
from .classics import run_hybrid_precomputed, stream_summary
from .replay_scenarios import build_replay_scenario

ZDATA = round3.DATA


def _case_stats(case):
    ts = [a.ts for a in case]
    fams = Counter(a.family for a in case)
    srcs = Counter(a.src for a in case)
    dsts = Counter(a.dst for a in case)
    return dict(
        size=len(case),
        span_min=round((max(ts) - min(ts)) / 60.0, 1),
        n_srcs=len(srcs), n_dsts=len(dsts),
        n_families=len(fams),
        top_family=fams.most_common(1)[0][0],
        top_src=srcs.most_common(1)[0][0],
        incidents=sorted({a.gt_incident for a in case if a.is_attack}),
    )


def arm_partition(sc, arm, g=None):
    """Return the arm's case partition for the scenario."""
    alerts = sc.alerts
    if arm == "incumbent":
        return arms.run_static(alerts, dims=["src", "dst"])["cases"]
    if arm == "family":
        return arms.run_static(alerts, dims=["family"])["cases"]
    if arm == "src_family":
        return arms.run_static(alerts, dims=["src", "family"])["cases"]
    if arm == "best_fixed":
        best, best_ari = None, -1.0
        for _label, dims in classics.DIM_SETS:
            r = arms.run_static(alerts, dims=dims)
            ev = metrics.evaluate(r["cases"], len(alerts), len(sc.incidents))
            if ev["ari"] > best_ari:
                best, best_ari = r, ev["ari"]
        return best["cases"]
    if arm == "psh":
        return arms.run_hybrid(alerts)["cases"]
    if arm == "julisch":
        return classics.run_julisch(alerts)["cases"]
    if arm == "oracle":
        og = classics.OracleGen()
        return og.run(alerts)
    if arm == "llm_c1":
        from .locals import _LLM_USER
        pols = []
        for w in _windowize(alerts, 300):
            block = json.dumps(stream_summary(w, level=1), indent=1)
            pols.append(g.generate_policy_text(
                _LLM_USER.format(summary=block), n_window=len(w)))
        return run_hybrid_precomputed(alerts, pols)["cases"]
    raise ValueError(arm)


def card(sc, ref_incidents, arms_list, g=None, arm_filter=None):
    """One card: for the reference incident set, the cases each arm emits
    that touch it. ref_incidents: set of gt_incident strings forming the
    reference (a single incident, or the D2 twin pair)."""
    ref = [a for a in sc.alerts if a.gt_incident in ref_incidents]
    out = dict(scenario=sc.name, n_ref_alerts=len(ref),
               ref_incidents=sorted(ref_incidents), arms={})
    n = len(sc.alerts)
    for arm in arms_list:
        if arm_filter and arm not in arm_filter:
            continue
        t0 = time.perf_counter()
        part = arm_partition(sc, arm, g=g)
        ev = metrics.evaluate(part, n, len(sc.incidents))
        touching = [c for c in part
                    if any(a.gt_incident in ref_incidents for a in c)]
        ref_only = [c for c in part
                    if all(a.gt_incident in ref_incidents
                           or not a.is_attack for a in c)]
        cases = sorted((_case_stats(c) for c in touching),
                       key=lambda s: -s["size"])
        out["arms"][arm] = dict(
            ari=ev["ari"], n_cases_total=len(part),
            n_cases_touching_ref=len(touching),
            n_cases_only_ref=len(ref_only),
            ref_split=len(touching) > 1,
            over_merge=any(len({a.gt_incident for a in c if a.is_attack}
                               & ref_incidents) < len(
                                 {a.gt_incident for a in c if a.is_attack})
                           for c in touching),
            cases=cases[:12],
            secs=round(time.perf_counter() - t0, 1))
        print(f"  [{sc.name} {arm}] ari={ev['ari']:.3f} "
              f"cases(total/touch/only-ref)={len(part)}/"
              f"{len(touching)}/{len(ref_only)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/case_cards.json")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    import torch
    from .locals import LocalLLMPolicyGen

    g = None
    if not args.no_llm:
        from .round3 import QWEN
        g = LocalLLMPolicyGen(QWEN, device=args.device,
                              dtype=torch.float16)

    llm_arms = ["llm_c1"] if g else []
    cards = {}

    # --- A2: Wednesday, reference = the DoS Hulk incident --------------
    a2 = scenarios.build_scenario(ZDATA, "A2", seed=17)
    cards["A2"] = card(a2, {"dos-hulk"},
                       ["incumbent", "family", "psh", "oracle"] + llm_arms, g)

    # --- D2: the twin pair ---------------------------------------------
    d2 = scenarios.build_twins(ZDATA, seed=17)
    cards["D2"] = card(d2, {"dos-hulk", "dos-hulk-twin"},
                       ["incumbent", "family", "src_family", "oracle"]
                       + llm_arms, g)

    # --- E: Thursday, reference = the 3-phase infiltration incident ---
    e = scenarios.build_scenario(ZDATA, "E", seed=17)
    cards["E"] = card(e, {"infiltration"},
                      ["incumbent", "best_fixed", "psh", "oracle"]
                      + llm_arms, g)

    # --- R-Thursday: the replayed C2 burst -----------------------------
    from .replay_scenarios import DEFAULT_EVE_DIR
    rth = build_replay_scenario("thursday", DEFAULT_EVE_DIR)
    cards["R-thursday"] = card(
        rth, {"infiltration"},
        ["incumbent", "family", "julisch", "oracle"] + llm_arms, g)

    with open(args.out, "w") as f:
        json.dump(cards, f, indent=1)
    print(f"case cards -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
