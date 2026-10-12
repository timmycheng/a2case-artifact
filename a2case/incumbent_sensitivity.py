"""P1-R + P1-M + P1-O (Omar revision brief 2026-10-08, deterministic legs).

- P1-R incumbent sensitivity: family and (family,dst) static arms on the
  full 36-instance parametric suite, plus paired tests of the stored
  deployed LLM arm (hybrid_llm_0.5B, round3_local + round3_expansion)
  against each, on the locked pooled-24 base and the single-16/mixed-20
  subgroup bases of tab:subgroup.
- P1-M complementary metric: B-Cubed F1 (attack-alert restriction, frozen
  definition in prereg_contract_n20.md) computed for the deterministic arms
  (incumbent rerun, family, family,dst, hindsight-best fixed, PSH, oracle)
  on the mixed-20 suite; the LLM S1/S3 rows get theirs in-batch from
  contract_gap_n20.
- P1-O: one-column oracle trace on D2 (window, dims, window_min,
  global_ari) dumped for the appendix table.

Everything here is deterministic (no GPU, no LLM calls). The incumbent
rerun doubles as an instance-reconstruction identity check against the
stored round-3 static_incumbent rows; the best-fixed rerun checks against
static_best_fixed_hindsight.

Entry (from experiments/):
    python -m a2case.incumbent_sensitivity --data <csv-dir> \
        [--out results/incumbent_sensitivity.json] \
        [--leg static36|mixed20|d2trace|analyze]
"""

import argparse
import json
import os
import time

from . import arms, metrics, round3, scenarios
from .classics import DIM_SETS, OracleGen

MIXED20 = (["P05", "P06", "P11", "P12", "P17", "P18", "P23", "P24"]
           + [f"P{p:02d}" for p in range(25, 37)])
ALL36 = [f"P{p:02d}" for p in range(1, 37)]
SINGLE16 = [f"P{p:02d}" for p in range(1, 25)
            if ((p - 1) % 6) not in (4, 5)]
N_BOOT = 20_000

from .contract_gap_n20 import bcubed  # frozen definition


def _atomic_dump(rows, path, tries=8):
    for i in range(tries):
        try:
            with open(path, "w") as f:
                json.dump(rows, f, indent=1)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


SENS_DIMS = [("static_family", ["family"]),
             ("static_family_dst", ["family", "dst"])]


def _build(zf, pid, seed=17):
    """Unified parametric builder: P01-P24 via round3, P25-P36 via the
    pre-registered expansion builder (same seed semantics)."""
    if pid <= 24:
        return round3.build_parametric(zf, pid, seed=seed)
    from .mixed_expansion import build_expansion
    return build_expansion(zf, pid, seed=seed)


def _eval_row(part, scenario, arm, res, sc):
    n = len(sc.alerts)
    ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
    return dict(part=part, scenario=scenario, arm=arm, seed=17,
                n_alerts=n, n_incidents=len(sc.incidents), **ev,
                **bcubed(res["cases"]))


def _load_for(out_path, part, need_arms):
    """Load all rows; done = scenarios of `part` with a complete arm set.
    Part-scoped so one leg's rows never mark another leg's instances done
    (the first mixed20 pass silently skipped everything this way)."""
    from collections import Counter
    rows = json.load(open(out_path)) if os.path.exists(out_path) else []
    cnt = Counter(r["scenario"] for r in rows if r["part"] == part)
    done = {s for s, c in cnt.items() if c >= need_arms}
    return rows, done


def _drop_partial(rows, part, scenario):
    return [r for r in rows
            if not (r["part"] == part and r["scenario"] == scenario)]


def run_static36(zf, out_path):
    rows, done = _load_for(out_path, "inc_sens_static", 3)
    for pid_str in ALL36:
        if pid_str in done:
            continue
        pid = int(pid_str[1:])
        sc = _build(zf, pid, seed=17)
        rows = _drop_partial(rows, "inc_sens_static", pid_str)
        jobs = ([("static_incumbent_rerun",
                  arms.run_static(sc.alerts, dims=["src", "dst"]))]
                + [(name, arms.run_static(sc.alerts, dims=dims))
                   for name, dims in SENS_DIMS])
        for name, res in jobs:
            rows.append(_eval_row("inc_sens_static", pid_str, name, res, sc))
        _atomic_dump(rows, out_path)
        line = " ".join(f"{r['arm']}={r['ari']:.3f}"
                        for r in rows[-3:])
        print(f"  [{pid_str}] {line}", flush=True)
    print(f"static36 leg complete: {len(rows)} rows total", flush=True)


def run_mixed20(zf, out_path):
    """Best-fixed sweep (7 sets @ w=10, the paper's hindsight arm) + the
    full 14-candidate best-fixed (7 sets x w in {5,10}; definitional check
    for the boxed candidate-space paragraph) + PSH + oracle on the
    mixed-20 (B-Cubed scope)."""
    rows, done = _load_for(out_path, "inc_sens_mixed", 4)
    for pid_str in MIXED20:
        if pid_str in done:
            continue
        pid = int(pid_str[1:])
        sc = _build(zf, pid, seed=17)
        assert sc.stats["twin"] is True
        rows = _drop_partial(rows, "inc_sens_mixed", pid_str)
        n = len(sc.alerts)
        best, best_ari = None, -1.0
        for label, dims in DIM_SETS:
            r = arms.run_static(sc.alerts, dims=dims)
            ev = metrics.evaluate(r["cases"], n, len(sc.incidents))
            if ev["ari"] > best_ari:
                best, best_ari = r, ev["ari"]
        rows.append(_eval_row("inc_sens_mixed", pid_str,
                              "static_best_fixed_rerun", best, sc))
        best14, best14_ari = None, -1.0
        for label, dims in DIM_SETS:
            for wmin in (5, 10):
                r = arms.run_static(sc.alerts, dims=dims, window_min=wmin)
                ev = metrics.evaluate(r["cases"], n, len(sc.incidents))
                if ev["ari"] > best14_ari:
                    best14, best14_ari = r, ev["ari"]
        rows.append(_eval_row("inc_sens_mixed", pid_str,
                              "static_best_fixed_w14", best14, sc))
        rows.append(_eval_row("inc_sens_mixed", pid_str,
                              "hybrid_psh_rerun",
                              arms.run_hybrid(sc.alerts), sc))
        og = OracleGen()
        t0 = time.perf_counter()
        part = og.run(sc.alerts)
        oracle_res = dict(cases=part, wall=time.perf_counter() - t0,
                          policy_calls=og.calls, prompt_tokens=0,
                          completion_tokens=0, fallbacks=0,
                          policy_trace=og.trace)
        rows.append(_eval_row("inc_sens_mixed", pid_str,
                              "oracle_rerun", oracle_res, sc))
        _atomic_dump(rows, out_path)
        print(f"  [{pid_str}] bf={rows[-4]['ari']:.4f} "
              f"bf14={rows[-3]['ari']:.4f} "
              f"psh={rows[-2]['ari']:.4f} oracle={rows[-1]['ari']:.4f}",
              flush=True)
    print(f"mixed20 leg complete: {len(rows)} rows total", flush=True)


def run_d2trace(zf, out_path="results/oracle_d2_trace.json"):
    sc = scenarios.build_twins(zf, seed=17)
    og = OracleGen()
    part = og.run(sc.alerts)
    ev = metrics.evaluate(part, len(sc.alerts), len(sc.incidents))
    with open(out_path, "w") as f:
        json.dump(dict(scenario="D2", seed=17, final_ari=ev["ari"],
                       n_windows=len(og.trace), trace=og.trace), f, indent=1)
    print(f"D2 oracle trace: {len(og.trace)} windows, final ARI "
          f"{ev['ari']:.4f} (expect 1.000) -> {out_path}", flush=True)


def _load_llm():
    llm = {}
    for path in ("results/round3_local.json", "results/round3_expansion.json"):
        for r in json.load(open(path)):
            if r["part"] == "parametric":
                llm.setdefault(r["arm"], {})[r["scenario"]] = r["ari"]
    return llm


def _boot_ci(d, n_boot=N_BOOT, seed=17):
    import numpy as np
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    boots = [float(np.mean(rng.choice(d, size=len(d), replace=True)))
             for _ in range(n_boot)]
    return np.percentile(boots, [2.5, 97.5])


def analyze(out_path="results/incumbent_sensitivity.json"):
    import numpy as np
    from scipy import stats as st

    rows = json.load(open(out_path))
    st_ = {(r["scenario"], r["arm"]): r
           for r in rows if r["part"] == "inc_sens_static"}
    mx = {(r["scenario"], r["arm"]): r
          for r in rows if r["part"] == "inc_sens_mixed"}
    llm = _load_llm()

    # identity checks: incumbent rerun and best-fixed rerun vs stored
    bad = []
    for pid_str in ALL36:
        r = st_.get((pid_str, "static_incumbent_rerun"))
        s = llm.get("static_incumbent", {}).get(pid_str)
        if r and s is not None and abs(r["ari"] - s) > 1e-12:
            bad.append((pid_str, "incumbent", r["ari"], s))
    for pid_str in MIXED20:
        r = mx.get((pid_str, "static_best_fixed_rerun"))
        s = llm.get("static_best_fixed_hindsight", {}).get(pid_str)
        if r and s is not None and abs(r["ari"] - s) > 1e-12:
            bad.append((pid_str, "best_fixed", r["ari"], s))
    print("identity checks vs stored round-3 rows: "
          + ("ALL BIT-IDENTICAL" if not bad else f"MISMATCH {bad}"))

    bases = [("pooled-24 (locked suite)", [f"P{p:02d}" for p in range(1, 25)]),
             ("single-16", SINGLE16),
             ("mixed-20", MIXED20)]
    for sens_name in ("static_family", "static_family_dst"):
        print(f"\n=== LLM (deployed S1) vs {sens_name} ===")
        for base_name, insts in bases:
            a = [llm["hybrid_llm_0.5B"][s] for s in insts]
            b = [st_[(s, sens_name)]["ari"] for s in insts]
            d = np.asarray(a) - np.asarray(b)
            nz = d[d != 0]
            msg = (f"{base_name:24s} dmean={d.mean():+.4f} "
                   f"W/T/L={int((d > 0).sum())}/{int((d == 0).sum())}/"
                   f"{int((d < 0).sum())}")
            if len(nz):
                w = st.wilcoxon(nz)
                lo, hi = _boot_ci(d)
                msg += f" p={w.pvalue:.4g} CI[{lo:+.4f},{hi:+.4f}]"
            print("  " + msg)
        m = float(np.mean([st_[(s, sens_name)]["ari"] for s in MIXED20]))
        m2 = float(np.mean([st_[(s, sens_name)]["ari"] for s in SINGLE16]))
        print(f"  means: single-16 {m2:.4f}  mixed-20 {m:.4f}")

    # B-Cubed summary on mixed-20 for every arm we hold (sensitivity arms
    # live in the static part, mixed-extras in the mixed part)
    print("\n=== B-Cubed F1 (attack restriction), mixed-20 ===")
    allrows = dict(st_)
    allrows.update(mx)
    arms_report = [("static_incumbent_rerun", "incumbent"),
                   ("static_family", "family"),
                   ("static_family_dst", "family,dst"),
                   ("static_best_fixed_rerun", "best-fixed (7@w10)"),
                   ("static_best_fixed_w14", "best-fixed (14)"),
                   ("hybrid_psh_rerun", "PSH"),
                   ("oracle_rerun", "oracle")]
    for key, label in arms_report:
        vals = [allrows[(s, key)]["bcubed_f"] for s in MIXED20
                if (s, key) in allrows]
        aris = [allrows[(s, key)]["ari"] for s in MIXED20
                if (s, key) in allrows]
        if vals:
            print(f"  {label:20s} ARI {np.mean(aris):.4f}  "
                  f"BCubedF1 {np.mean(vals):.4f}  (n={len(vals)})")
    bf = [mx[(s, "static_best_fixed_rerun")]["ari"] for s in MIXED20
          if (s, "static_best_fixed_rerun") in mx]
    bf14 = [mx[(s, "static_best_fixed_w14")]["ari"] for s in MIXED20
            if (s, "static_best_fixed_w14") in mx]
    if bf and bf14:
        d = np.asarray(bf14) - np.asarray(bf)
        print(f"  best-fixed 14 vs 7@w10: dmean={d.mean():+.4f} "
              f"changed {int((d > 1e-12).sum())}/{len(d)} instances")
    cg = "results/contract_gap_n20.json"
    if os.path.exists(cg):
        for lv, name in ((1, "LLM S1"), (3, "LLM S3")):
            vals = [r["bcubed_f"] for r in json.load(open(cg))
                    if r.get("part") == "contract_gap_n20" and r["level"] == lv]
            if vals:
                print(f"  {name:12s} BCubedF1 {np.mean(vals):.4f}  "
                      f"(n={len(vals)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="results/incumbent_sensitivity.json")
    ap.add_argument("--leg", choices=["static36", "mixed20", "d2trace",
                                      "analyze", "all"],
                    default="all")
    args = ap.parse_args()
    data = args.data or round3.DATA
    if args.leg in ("static36", "all"):
        run_static36(data, args.out)
    if args.leg in ("mixed20", "all"):
        run_mixed20(data, args.out)
    if args.leg in ("d2trace", "all"):
        run_d2trace(data)
    if args.leg in ("analyze", "all"):
        analyze(args.out)


if __name__ == "__main__":
    main()
