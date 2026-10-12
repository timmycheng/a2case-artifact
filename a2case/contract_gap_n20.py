"""P0-R (Omar revision brief 2026-10-08, P0 "Run S3 (and S1 control) on the
full mixed-regime n=20" + P1 "multiple seeds on the contract result").

S1-vs-S3 contract contrast on the pre-registered mixed-regime n=20 suite,
plus seed replication on D2 (seeds 17/29/43) and on P12/P24 (seeds 29/43;
seed 17 comes from the main batch). Preregistration:
experiments/prereg_contract_n20.md (frozen 2026-10-08 before any run).

Mirrors contract_gap.py; differences are confined to the instance set, the
stored per-window mechanism trace (dims + window_min + fallback flag +
twin-family/twin-source counts per window), the B-Cubed complementary
metric, and the seed legs.

Entry (from experiments/):
    python -m a2case.contract_gap_n20 --data <csv-dir> \
        [--out results/contract_gap_n20.json] [--device cuda] \
        [--leg mixed|seeds|all]
    python -m a2case.contract_gap_n20 --only P23 --device cpu   # anchor patch
    python -m a2case.contract_gap_n20 --analyze-only
"""

import argparse
import json
import os
import time

from . import metrics, round3, scenarios
from .arms import _windowize
from .classics import run_hybrid_precomputed, stream_summary
from .locals import LocalLLMPolicyGen

MIXED20 = (["P05", "P06", "P11", "P12", "P17", "P18", "P23", "P24"]
           + [f"P{p:02d}" for p in range(25, 37)])
TWIN_FAMILY = {"tuesday": "FTP-Patator", "wednesday": "DoS Hulk",
               "thursday": "Web Attack - Brute Force", "friday": "PortScan"}
TWIN_SRC = round3.TWIN_SRC
LEVELS = [1, 3]
N_BOOT = 20_000


def _atomic_dump(rows, path, tries=8):
    """Cloud-sync clients transiently lock result files (PermissionError 13
    observed mid-run in audit_sensitivity): retry with backoff."""
    for i in range(tries):
        try:
            with open(path, "w") as f:
                json.dump(rows, f, indent=1)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def bcubed(partition):
    """B-Cubed precision/recall/F1 with multiplicity (Amigo et al. 2009),
    computed on the attack-alert restriction -- identical restriction to
    metrics.evaluate's ARI. Frozen definition in the prereg."""
    from collections import Counter
    cls_total, clu_total, joint = Counter(), Counter(), Counter()
    for ci, case in enumerate(partition):
        for a in case:
            if a.is_attack:
                cls_total[a.gt_incident] += 1
                clu_total[ci] += 1
                joint[(ci, a.gt_incident)] += 1
    if not cls_total:
        return dict(bcubed_p=float("nan"), bcubed_r=float("nan"),
                    bcubed_f=float("nan"))
    p_sum = r_sum = n = 0.0
    for (ci, cls), m in joint.items():
        p_sum += m * (m / clu_total[ci])
        r_sum += m * (m / cls_total[cls])
        n += m
    p, r = p_sum / n, r_sum / n
    f = 2 * p * r / (p + r) if p + r else float("nan")
    return dict(bcubed_p=p, bcubed_r=r, bcubed_f=f)


def _window_stats(w, twin_family):
    if twin_family is None:
        return len(w), None, None
    n_fam = sum(1 for a in w if a.family == twin_family)
    n_src = sum(1 for a in w if a.src == TWIN_SRC)
    return len(w), n_fam, n_src


def _run_instance(g, sc, part, tag, seed, twin_family, rows, out_path,
                  deployment):
    from .locals import _LLM_USER
    from .policies import stream_summary as deployed_summary

    windows = _windowize(sc.alerts, 300)
    n = len(sc.alerts)
    if twin_family is not None:
        # deployed-equivalence guard (same as contract_gap): level-1 block
        # must be byte-identical to what arms.run_hybrid would have sent
        for w in windows[:5]:
            a = json.dumps(stream_summary(w, level=1), indent=1)
            b = json.dumps(deployed_summary(w), indent=1)
            assert a == b, "S1 block != deployed block -- abort"
    for level in LEVELS:
        pre = (g.prompt_tokens, g.completion_tokens, g.calls, g.fallbacks)
        pols = []
        for w in windows:
            block = json.dumps(stream_summary(w, level=level), indent=1)
            user = _LLM_USER.format(summary=block)
            pols.append(g.generate_policy_text(user, n_window=len(w)))
        res = run_hybrid_precomputed(sc.alerts, pols)
        ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
        bc = bcubed(res["cases"])
        ptok = g.prompt_tokens - pre[0]
        ctok = g.completion_tokens - pre[1]
        calls = g.calls - pre[2]
        nfb = g.fallbacks - pre[3]
        trace = []
        for wi, (w, p) in enumerate(zip(windows, pols)):
            nw, nfam, nsrc = _window_stats(w, twin_family)
            trace.append(dict(window=wi, dims=list(p.dims),
                              window_min=p.window_min, source=p.source,
                              n=nw, n_twin_family=nfam, n_twin_src=nsrc))
        rows.append(dict(
            part=part, scenario=tag, arm=f"qwen0.5B_S{level}", level=level,
            seed=seed, deployment=deployment, n_alerts=n,
            n_incidents=len(sc.incidents), n_windows=len(windows),
            **ev, **bc,
            llm_valid=calls - nfb, llm_calls=calls, fallbacks=nfb,
            prompt_tokens=ptok, completion_tokens=ctok,
            prompt_tokens_per_call=ptok / max(1, calls),
            policy_trace=trace,
            day=sc.stats.get("day")))
        _atomic_dump(rows, out_path)
        print(f"  [{tag} S{level}] ari={ev['ari']:.4f} "
              f"bcF1={bc['bcubed_f']:.4f} ptok/call={ptok / max(1, calls):.0f} "
              f"fb={nfb}/{calls}", flush=True)


def _load_refs():
    """Stored round-3 reference ARIs for the mixed-20 (anchor + oracle +
    best-fixed), from round3_local (P05-P24) and round3_expansion
    (P25-P36)."""
    ref = {}
    for path in ("results/round3_local.json", "results/round3_expansion.json"):
        for r in json.load(open(path)):
            if r["part"] == "parametric" and r["scenario"] in MIXED20:
                ref.setdefault(r["arm"], {})[r["scenario"]] = r["ari"]
    return ref


def _build(zf, pid, seed=17):
    """Unified parametric builder: P01-P24 via round3, P25-P36 via the
    pre-registered expansion builder (same seed semantics)."""
    if pid <= 24:
        return round3.build_parametric(zf, pid, seed=seed)
    from .mixed_expansion import build_expansion
    return build_expansion(zf, pid, seed=seed)


def _deployment_tag(device):
    return f"{'fp16' if device != 'cpu' else 'fp32'}-{device}"


def run_mixed(zf, out_path, device="cuda", only=None):
    import torch

    from .round3 import QWEN as qwen_path

    rows = []
    if os.path.exists(out_path):
        rows = [r for r in json.load(open(out_path))
                if not (r["part"] == "contract_gap_n20"
                        and r["scenario"] in (only or []))]
    from collections import Counter
    lv_cnt = Counter(r["scenario"] for r in rows
                     if r["part"] == "contract_gap_n20")
    done = {s for s, c in lv_cnt.items() if c >= len(LEVELS)}
    refs = _load_refs()
    g = LocalLLMPolicyGen(qwen_path, device=device, dtype=torch.float16)
    dep = _deployment_tag(device)
    for pid_str in MIXED20:
        if pid_str in done or (only and pid_str not in only):
            continue
        pid = int(pid_str[1:])
        sc = _build(zf, pid, seed=17)
        assert sc.stats["twin"] is True, f"{pid_str} not a twin instance"
        rows = [r for r in rows if r["scenario"] != pid_str]  # drop partial
        day = sc.stats["day"]
        _run_instance(g, sc, "contract_gap_n20", pid_str, 17,
                      TWIN_FAMILY[day], rows, out_path, dep)
        a1 = [r for r in rows if r["part"] == "contract_gap_n20"
              and r["scenario"] == pid_str and r["level"] == 1][0]["ari"]
        r3s1 = refs.get("hybrid_llm_0.5B", {}).get(pid_str)
        flag = ("ok" if r3s1 is not None and abs(a1 - r3s1) < 1e-12
                else "DIFF/missing")
        print(f"  [{pid_str}] S1 anchor vs round3: {flag} "
              f"(new {a1:.6f} vs stored {r3s1})", flush=True)
    print(f"mixed n=20 leg complete: {len(rows)} rows total", flush=True)


def run_seeds(zf, out_path, device="cuda"):
    import torch

    from .round3 import QWEN as qwen_path

    rows = []
    if os.path.exists(out_path):
        rows = [r for r in json.load(open(out_path))
                if r["part"] != "contract_seeds"]
    from collections import Counter
    lv_cnt = Counter(r["scenario"] for r in rows
                     if r["part"] == "contract_seeds")
    done = {s for s, c in lv_cnt.items() if c >= len(LEVELS)}
    days = ["tuesday", "wednesday", "thursday", "friday"]
    jobs = ([("D2", s, lambda s=s: scenarios.build_twins(zf, seed=s),
              "DoS Hulk") for s in (17, 29, 43)]
            + [(f"P{p:02d}", s,
                lambda p=p, s=s: round3.build_parametric(zf, p, seed=s),
                TWIN_FAMILY[days[(p - 1) // 6]])
               for p in (12, 24) for s in (29, 43)])
    g = LocalLLMPolicyGen(qwen_path, device=device, dtype=torch.float16)
    dep = _deployment_tag(device)
    for name, seed, build, fam in jobs:
        tag = f"{name}-s{seed}"
        if tag in done:
            print(f"skip {tag} (done)", flush=True)
            continue
        rows = [r for r in rows if r["scenario"] != tag]  # drop partial
        sc = build()
        if name != "D2":
            assert sc.stats["twin"] is True
        _run_instance(g, sc, "contract_seeds", tag, seed, fam, rows,
                      out_path, dep)
    print(f"seeds leg complete: {len(rows)} rows total", flush=True)


def _boot_ci(d, n_boot=N_BOOT, seed=17):
    import numpy as np
    rng = np.random.default_rng(seed)
    d = np.asarray(d)
    boots = [float(np.mean(rng.choice(d, size=len(d), replace=True)))
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return lo, hi


def _paired(name, a, b):
    """a - b paired test per the paper convention / prereg."""
    import numpy as np
    from scipy import stats as st

    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    line = (f"{name}: dmean={d.mean():+.4f} "
            f"wins/ties/losses={int((d > 0).sum())}/{int((d == 0).sum())}/"
            f"{int((d < 0).sum())}")
    if len(nz):
        w = st.wilcoxon(nz)
        lo, hi = _boot_ci(d)
        line += (f" wilcoxon p={w.pvalue:.4g} 95%CI[{lo:+.4f},{hi:+.4f}]")
    else:
        line += " all-zero differences"
    print("  " + line, flush=True)
    return d


def analyze(out_path="results/contract_gap_n20.json"):
    import numpy as np

    rows = json.load(open(out_path))
    mixed = {(r["scenario"], r["level"]): r
             for r in rows if r["part"] == "contract_gap_n20"}
    refs = _load_refs()
    missing = [s for s in MIXED20 if (s, 1) not in mixed or (s, 3) not in mixed]
    if missing:
        print(f"WARNING: mixed leg incomplete, missing {missing}")

    print(f"{'inst':6s} {'S1':>7s} {'S3':>7s} {'d':>7s} {'r3_S1':>7s} "
          f"{'anchor':>8s} {'oracle':>7s} {'bf':>7s}")
    s1, s3, orc, bf, anchors_ok = [], [], [], [], True
    for s in MIXED20:
        if (s, 1) not in mixed or (s, 3) not in mixed:
            continue
        a1, a3 = mixed[(s, 1)]["ari"], mixed[(s, 3)]["ari"]
        r3s1 = refs.get("hybrid_llm_0.5B", {}).get(s)
        o = refs.get("oracle", {}).get(s)
        b = refs.get("static_best_fixed_hindsight", {}).get(s)
        same = r3s1 is not None and abs(a1 - r3s1) < 1e-12
        anchors_ok &= same
        s1.append(a1), s3.append(a3), orc.append(o), bf.append(b)
        print(f"{s:6s} {a1:7.4f} {a3:7.4f} {a3 - a1:+7.4f} {r3s1:7.4f} "
              f"{'ok' if same else 'DIFF':>8s} {o:7.4f} {b:7.4f}")
    s1, s3, orc, bf = map(np.array, (s1, s3, orc, bf))
    n = len(s1)
    print(f"\nS1 anchor vs round3 rows ({n} instances): "
          f"{'ALL BIT-IDENTICAL' if anchors_ok else 'MISMATCH (patch needed: --only <pid> --device cpu)'}")
    print(f"mean ARI: S1={s1.mean():.4f}  S3={s3.mean():.4f}  "
          f"oracle={orc.mean():.4f}  best-fixed={bf.mean():.4f}")
    gap1 = float((orc - s1).mean())
    gap3 = float((orc - s3).mean())
    print(f"gap to oracle: S1={gap1:+.4f}  S3={gap3:+.4f}  "
          f"recovered fraction={(gap1 - gap3) / max(1e-12, gap1):.3f}  (n={n})")
    print("\npaired tests (prereg: two-sided Wilcoxon zero-dropped, "
          "bootstrap 20k rng17):")
    _paired("S3-S1 ", s3, s1)
    _paired("oracle-S3", orc, s3)
    _paired("oracle-S1", orc, s1)
    b1 = np.array([mixed[(s, 1)]["bcubed_f"] for s in MIXED20
                   if (s, 1) in mixed])
    b3 = np.array([mixed[(s, 3)]["bcubed_f"] for s in MIXED20
                   if (s, 3) in mixed])
    if len(b1) == len(b3) == n:
        print(f"B-Cubed F1 means (attack restriction): S1={b1.mean():.4f}  "
              f"S3={b3.mean():.4f}")

    # mechanism counts (prereg rule: twin-family fraction >= 0.5)
    print("\nmechanism counts, S3 on n=20 (window twin-family frac >= 0.5 "
          "=> twin-dominated):")
    for label, key in (("twin-dominated", True), ("noise-dominated", False)):
        c = {"family,src": 0, "family,dst": 0, "other": 0}
        for s in MIXED20:
            r = mixed.get((s, 3))
            if not r:
                continue
            for t in r["policy_trace"]:
                if t["n_twin_family"] is None:
                    continue
                twin_dom = t["n_twin_family"] / t["n"] >= 0.5
                if twin_dom != key:
                    continue
                d = ",".join(t["dims"])
                c["family,src" if d == "family,src"
                  else "family,dst" if d == "family,dst" else "other"] += 1
        tot = sum(c.values())
        print(f"  {label:15s} n={tot:4d}  " +
              "  ".join(f"{k}={v}" for k, v in c.items()))

    # seed replication table
    seeds = [r for r in rows if r["part"] == "contract_seeds"]
    if seeds:
        print("\nseed replication (greedy decode deterministic per stream; "
              "variation = stream seed):")
        tags = sorted({r["scenario"] for r in seeds})
        print(f"{'stream':10s} {'S1':>7s} {'S3':>7s}")
        for tag in tags:
            rs = {x["level"]: x["ari"] for x in seeds if x["scenario"] == tag}
            if 1 in rs and 3 in rs:
                print(f"{tag:10s} {rs[1]:7.4f} {rs[3]:7.4f}")
        for base in ("D2", "P12", "P24"):
            st_ = sorted(x for x in tags if x.startswith(base + "-s"))
            if st_:
                a3 = [x["ari"] for x in seeds
                      if x["scenario"] in st_ and x["level"] == 3]
                a1_ = [x["ari"] for x in seeds
                       if x["scenario"] in st_ and x["level"] == 1]
                print(f"  {base} spread: S1 [{min(a1_):.4f},{max(a1_):.4f}]  "
                      f"S3 [{min(a3):.4f},{max(a3):.4f}] over seeds "
                      f"{[t.split('-s')[1] for t in st_]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="results/contract_gap_n20.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--leg", choices=["mixed", "seeds", "all"],
                    default="all")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict the mixed leg to these instance ids "
                         "(anchor patch runs; CPU/fp32 by design)")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    data = args.data or round3.DATA
    if not args.analyze_only:
        if args.only and args.device != "cpu":
            raise SystemExit("--only patch runs are CPU/fp32 by design "
                             "(the GPU deployment diverged there)")
        if args.leg in ("mixed", "all"):
            run_mixed(data, args.out, device=args.device, only=args.only)
        if args.leg in ("seeds", "all"):
            run_seeds(data, args.out, device=args.device)
    analyze(args.out)


if __name__ == "__main__":
    main()
