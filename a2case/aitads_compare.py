"""CIC <-> AIT-ADS comparison tables and paired statistics.

Reads the AIT-ADS matrix (fast arms), the per-scenario LLM-0.5B records, and
the CIC round-3 results; produces:
  1. per-arm mean ARI table (CIC 24 parametric instances vs AIT-ADS 8
     scenario instances)
  2. paired tests vs the deployed incumbent (primary Gate 1 criterion) and
     vs best-fixed-hindsight (ceiling relation)
  3. the conclusion-by-conclusion comparison that feeds Phase 2 writing

Run: python -m a2i.aitads_compare
"""
import glob
import json
import os

from scipy import stats as st

RES = os.path.join(os.path.dirname(__file__), "..", "results", "gate0")
CIC = os.path.join(os.path.dirname(__file__), "..", "results",
                   "round3_local.json")
P24 = [f"P{i:02d}" for i in range(1, 25)]


def load_aitads():
    rows = json.load(open(os.path.join(RES, "aitads_matrix.json")))["rows"]
    out = {}
    for r in rows:
        out[(r["arm"], r["dimset"], r["scenario"])] = r["ari"]
    for f in glob.glob(os.path.join(RES, "llm05", "*.json")):
        r = json.load(open(f))
        out[("hybrid-LLM0.5B", "per-window", r["scenario"])] = r["ari"]
    return out


def load_cic():
    d = json.load(open(CIC))
    out = {}
    for r in d:
        if r.get("part") == "parametric" and r.get("seed") == 17:
            out[(r["arm"], r["scenario"])] = r["ari"]
    return out


def load_cic_family_sweep():
    """CIC family-key ARI from the benchmark static sweep (A1,A2,A3,B2,B3,E
    -- the circularity-ceiling evidence; the parametric set has no family
    row)."""
    path = os.path.join(os.path.dirname(__file__), "..", "results",
                        "static_sweep.json")
    rows = json.load(open(path))
    return {r["scenario"]: r["ari"] for r in rows if r["dimset"] == "family"}


def paired(ma, mb, keys, label):
    diffs = [ma[k] - mb[k] for k in keys]
    nz = [x for x in diffs if x != 0]
    p = st.wilcoxon(nz).pvalue if len(nz) >= 3 and len(set(nz)) > 1 \
        else float("nan")
    print(f"  {label:36s} mean_d={sum(diffs)/len(diffs):+.3f} "
          f"pos/neg/tie={sum(x > 0 for x in diffs)}/"
          f"{sum(x < 0 for x in diffs)}/{sum(x == 0 for x in diffs)} "
          f"wilcoxon p={p:.4f}")


def main():
    A = load_aitads()
    C = load_cic()
    ait_scen = sorted({k[2] for k in A})
    info = [s for s in ait_scen
            if sum(1 for k in A if k[2] == s and k[1] != "per-window"
                   and k[0] == "static") > 0 and s != "shaw"]
    print(f"AIT-ADS instances: {len(ait_scen)} scenarios "
          f"(n={len(info)} informative; shaw structurally degenerate)")

    def arm_mean_aitads(arm, dimset, keys):
        vals = [A.get((arm, dimset, k)) for k in keys]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    def arm_mean_cic(arm, keys=P24):
        vals = [C.get((arm, k)) for k in keys]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n== mean ARI: CIC vs AIT-ADS ==")
    print(f"  {'arm':28s} {'CIC':>7s} {'AIT-ADS':>8s}")
    for label, cic_arm, ait_arm, ait_dim in (
            ("incumbent (src,dst)", "static_incumbent", "static", "src,dst"),
            ("embedding (w2v)", "embedding_w2v", "embedding", "w2v-tau.75"),
            ("per-window PSH", "hybrid_psh", "hybrid-PSH", "per-window"),
            ("per-window LLM 0.5B", "hybrid_llm_0.5B", "hybrid-LLM0.5B",
             "per-window"),
            ("best-fixed-hindsight", "static_best_fixed_hindsight", None,
             None),
            ("oracle", "oracle", None, None)):
        cv = arm_mean_cic(cic_arm) if cic_arm else float("nan")
        av = (arm_mean_aitads(ait_arm, ait_dim, info)
              if ait_arm else float("nan"))
        print(f"  {label:28s} {cv:7.3f} {av:8.3f}")
    fam_cic = load_cic_family_sweep()
    print(f"  {'family (CIC: benchmark sweep)':28s} "
          f"{sum(fam_cic.values())/len(fam_cic):7.3f} "
          f"{arm_mean_aitads('static', 'family', info):8.3f}"
          f"   <- CIC ceiling 1.000 everywhere; AIT-ADS breaks it")

    def ait_map(arm, dimset):
        return {k: v for (a, d, k), v in A.items()
                if a == arm and d == dimset}

    def cic_map(arm):
        return {k: v for (a, k), v in C.items() if a == arm}

    print("\n== paired vs incumbent (Gate 1 primary criterion) ==")
    inc_a = ait_map("static", "src,dst")
    for arm, dimset in (("hybrid-LLM0.5B", "per-window"),
                        ("hybrid-PSH", "per-window"),
                        ("embedding", "w2v-tau.75"),
                        ("static", "family")):
        m = ait_map(arm, dimset)
        keys = [s for s in info if s in m]
        if not keys:
            continue
        paired(m, inc_a, keys, f"AIT-ADS {arm} - incumbent(src,dst)")

    # deployment-baseline set per architecture.tex: QRadar/Splunk = (src,dst),
    # Sentinel = sid (rule-keyed). Report vs sid as the second deployed key.
    sid_a = ait_map("static", "sid")
    for arm, dimset in (("hybrid-LLM0.5B", "per-window"),
                        ("hybrid-PSH", "per-window")):
        m = ait_map(arm, dimset)
        keys = [s for s in info if s in m and s in sid_a]
        if keys:
            paired(m, sid_a, keys, f"AIT-ADS {arm} - Sentinel(sid)")

    # hindsight-best fixed (per-scenario max over the 7 dim sets): the
    # single-regime clause of the paper's confirmatory structure expects
    # per-window selection to RECOVER this on single-regime streams.
    dim_sets = [d for d in {k[1] for k in A if k[0] == "static"}]
    hindsight = {}
    for s in info:
        vals = [A[("static", d, s)] for d in dim_sets if ("static", d, s) in A]
        if vals:
            hindsight[s] = max(vals)
    for arm, dimset in (("hybrid-LLM0.5B", "per-window"),):
        m = ait_map(arm, dimset)
        keys = [s for s in info if s in m]
        if keys:
            paired(m, hindsight, keys,
                   f"AIT-ADS {arm} - hindsight-best-fixed")

    cic_inc = cic_map("static_incumbent")
    for arm in ("hybrid_llm_0.5B", "hybrid_psh", "embedding_w2v"):
        m = cic_map(arm)
        paired(m, cic_inc, P24, f"CIC    {arm} - incumbent")

    print("\n== AIT-ADS per-scenario ARI (primary arms) ==")
    hdr = f"  {'arm':>22s}" + "".join(f"{s[:9]:>10s}" for s in ait_scen)
    print(hdr)
    for arm, dimset in (("static", "src,dst"), ("static", "family"),
                        ("embedding", "w2v-tau.75"),
                        ("hybrid-PSH", "per-window"),
                        ("hybrid-LLM0.5B", "per-window")):
        m = ait_map(arm, dimset)
        if not m:
            continue
        print(f"  {arm + '/' + dimset[:14]:>22s}"
              + "".join(f"{m.get(s, float('nan')):10.3f}" for s in ait_scen))


if __name__ == "__main__":
    main()
