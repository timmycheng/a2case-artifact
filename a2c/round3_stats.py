"""Paired statistics over the parametric suite (round3_local.json)."""

import json
import sys
from collections import defaultdict

import numpy as np
from scipy import stats


def boot_ci(diff, n=10000, seed=17):
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(diff, size=len(diff), replace=True)))
             for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(path="results/round3_local.json"):
    rows = json.load(open(path))
    inst = defaultdict(dict)
    for r in rows:
        if r["part"] == "parametric":
            inst[r["arm"]][r["scenario"]] = r["ari"]
    arms = list(inst.keys())
    common = sorted(set.intersection(*[set(v.keys()) for v in inst.values()]))
    print(f"paired instances: {len(common)}; arms: {arms}\n")

    print("=== per-arm summary over instances ===")
    for a in arms:
        v = np.array([inst[a][s] for s in common])
        print(f"  {a:30s} mean={v.mean():.3f} median={np.median(v):.3f} "
              f"std={v.std(ddof=1):.3f}")

    print("\n=== paired comparisons (ARI differences, Wilcoxon signed-rank) ===")
    def paired(a, b):
        d = np.array([inst[a][s] - inst[b][s] for s in common])
        d_nz = d[d != 0]
        if len(d_nz) == 0:
            return d.mean(), (0, 1), (0, 0), d
        w = stats.wilcoxon(d_nz)
        lo, hi = boot_ci(d)
        return d.mean(), (w.statistic, w.pvalue), (lo, hi), d
    pairs = [
        ("hybrid_psh", "static_incumbent"),
        ("hybrid_psh", "static_best_fixed_hindsight"),
        ("hybrid_llm_0.5B", "static_incumbent"),
        ("hybrid_llm_0.5B", "static_best_fixed_hindsight"),
        ("hybrid_llm_0.5B", "hybrid_psh"),
        ("oracle", "hybrid_llm_0.5B"),
        ("oracle", "static_best_fixed_hindsight"),
        ("julisch", "static_incumbent"),
        ("embedding_w2v", "static_incumbent"),
    ]
    for a, b in pairs:
        if a not in inst or b not in inst:
            continue
        mean, (stat, p), (lo, hi), d = paired(a, b)
        print(f"  {a:26s} - {b:28s} dmean={mean:+.3f} "
              f"wilcoxon p={p:.4f} 95%CI[{lo:+.3f},{hi:+.3f}]")


# Parametric-suite regime split: CONFIGS[(pid-1) % 6] in round3.py —
# configs 4 and 5 carry twin=True (concurrency injection = mixed regime).
MIXED = [f"P{p:02d}" for p in range(1, 25) if ((p - 1) % 6) in (4, 5)]


def subgroup(path="results/round3_local.json"):
    """Regime subgroup analysis: single-regime (16) vs mixed-regime (8)."""
    rows = json.load(open(path))
    inst = defaultdict(dict)
    for r in rows:
        if r["part"] == "parametric":
            inst[r["arm"]][r["scenario"]] = r["ari"]
    single = [f"P{p:02d}" for p in range(1, 25) if f"P{p:02d}" not in MIXED]
    for name, sub in (("single-regime", single), ("mixed-regime", MIXED)):
        print(f"=== {name} (n={len(sub)}) ===")
        for a in ("static_incumbent", "static_best_fixed_hindsight",
                  "hybrid_psh", "hybrid_llm_0.5B", "oracle"):
            v = np.array([inst[a][s] for s in sub])
            print(f"  {a:30s} mean={v.mean():.3f} median={np.median(v):.3f}")
        for a, b in (("hybrid_llm_0.5B", "static_incumbent"),
                     ("hybrid_llm_0.5B", "static_best_fixed_hindsight"),
                     ("hybrid_psh", "static_incumbent"),
                     ("oracle", "static_best_fixed_hindsight")):
            d = np.array([inst[a][s] - inst[b][s] for s in sub])
            nz = d[d != 0]
            if len(nz) == 0:
                print(f"  {a} - {b}: all-zero diffs")
                continue
            w = stats.wilcoxon(nz)
            lo, hi = boot_ci(d)
            print(f"  {a:26s}-{b:28s} dmean={d.mean():+.3f} p={w.pvalue:.4f}"
                  f" 95%CI[{lo:+.3f},{hi:+.3f}] wins={int((d > 0).sum())}/{len(d)}")
    # regime x arm interaction (Mann-Whitney on per-instance paired diffs)
    for a, b, alt, label in (
            ("hybrid_llm_0.5B", "static_incumbent", "less",
             "LLM-incumbent larger on single"),
            ("static_best_fixed_hindsight", "hybrid_llm_0.5B", "greater",
             "bestfixed-LLM larger on mixed")):
        d_single = np.array([inst[a][s] - inst[b][s] for s in single])
        d_mixed = np.array([inst[a][s] - inst[b][s] for s in MIXED])
        u = stats.mannwhitneyu(d_mixed, d_single, alternative=alt)
        print(f"interaction [{label}]: MW p={u.pvalue:.4g}")


# Pre-registered expansion (prereg_mixed_expansion.md, 2026-09-21):
# instances P25-P36 are twin-on by construction; the expanded mixed set is
# the round-3 twin instances plus P25-P36 (n=20). Single set unchanged.
MIXED_EXPANDED = MIXED + [f"P{p:02d}" for p in range(25, 37)]


def subgroup_expanded(path="results/round3_local.json",
                      expansion_path="results/round3_expansion.json"):
    """The pre-registered single-batch analysis on the expanded mixed set
    (n=20): identical arm means, paired comparisons and interaction tests
    as subgroup(), computed once. Exploratory variants are not
    pre-approved; label any later additions exploratory in the paper."""
    inst = defaultdict(dict)
    for r in json.load(open(path)):
        if r["part"] == "parametric":
            inst[r["arm"]][r["scenario"]] = r["ari"]
    if expansion_path:
        for r in json.load(open(expansion_path)):
            if r["part"] == "parametric":
                inst[r["arm"]][r["scenario"]] = r["ari"]
    single = [f"P{p:02d}" for p in range(1, 25)
              if f"P{p:02d}" not in MIXED]
    missing = [s for s in MIXED_EXPANDED
               if any(s not in inst[a] for a in
                      ("static_incumbent", "hybrid_llm_0.5B", "oracle"))]
    if missing:
        print(f"expanded set incomplete, missing arms for: {missing}")
        return
    for name, sub in (("single-regime", single),
                      ("mixed-regime-expanded", MIXED_EXPANDED)):
        print(f"=== {name} (n={len(sub)}) ===")
        for a in ("static_incumbent", "static_best_fixed_hindsight",
                  "hybrid_psh", "hybrid_llm_0.5B", "oracle"):
            v = np.array([inst[a][s] for s in sub])
            print(f"  {a:30s} mean={v.mean():.3f} median={np.median(v):.3f}")
        for a, b in (("hybrid_llm_0.5B", "static_incumbent"),
                     ("hybrid_llm_0.5B", "static_best_fixed_hindsight"),
                     ("hybrid_psh", "static_incumbent"),
                     ("oracle", "static_best_fixed_hindsight")):
            d = np.array([inst[a][s] - inst[b][s] for s in sub])
            nz = d[d != 0]
            if len(nz) == 0:
                print(f"  {a} - {b}: all-zero diffs")
                continue
            w = stats.wilcoxon(nz)
            lo, hi = boot_ci(d)
            print(f"  {a:26s}-{b:28s} dmean={d.mean():+.3f} p={w.pvalue:.4f}"
                  f" 95%CI[{lo:+.3f},{hi:+.3f}] wins={int((d > 0).sum())}/{len(d)}")
    for a, b, alt, label in (
            ("hybrid_llm_0.5B", "static_incumbent", "less",
             "LLM-incumbent larger on single"),
            ("static_best_fixed_hindsight", "hybrid_llm_0.5B", "greater",
             "bestfixed-LLM larger on mixed")):
        d_single = np.array([inst[a][s] - inst[b][s] for s in single])
        d_mixed = np.array([inst[a][s] - inst[b][s] for s in MIXED_EXPANDED])
        u = stats.mannwhitneyu(d_mixed, d_single, alternative=alt)
        print(f"interaction [{label}]: MW p={u.pvalue:.4g}")


def holm(pvals):
    """Holm step-down adjusted p-values (reported order irrelevant)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj, prev = np.empty(m), 0.0
    for rank, i in enumerate(order):
        prev = max(prev, (m - rank) * pvals[i])
        adj[i] = min(prev, 1.0)
    return adj


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/round3_local.json")
    print()
    subgroup(sys.argv[1] if len(sys.argv) > 1 else "results/round3_local.json")
