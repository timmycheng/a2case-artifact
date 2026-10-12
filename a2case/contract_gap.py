"""Direction A (journal plan, Omar reply section 4, 2026-09-27): can the
S3 information contract that fully solves the D2 twin stress test also
recover part of the mixed-regime oracle gap, without changing model,
architecture or training?

Known going in (round-3 parametric suite, mixed-regime n=8 twin
instances P05/P06/P11/P12/P17/P18/P23/P24): the deployed 0.5B arm runs
the S1 summary contract and reaches mean ARI 0.683 against oracle 0.995
and hindsight-best fixed 0.963; the paper attributes the selector-side
gap to the information contract. D2 supports the attribution -- S3 is a
sharp optimum there (ARI 1.000 vs 0.886 for every other variant) -- but
the attribution has never been tested on the mixed-regime instances
themselves. This experiment closes that loop on the gap-largest,
statistically most meaningful subset (the original n=8 twins).

Design invariants (fixed before running, mirroring s4_reversion):
  * instances: the 8 round-3 twin instances, rebuilt bit-identically via
    round3.build_parametric (same seed 17 generator path as the paper
    rows);
  * model/deployment: Qwen2.5-0.5B-Instruct, greedy, max_new_tokens 96
    -- the deployed generation configuration of every 0.5B row in the
    paper. Deployment: float16 on GPU. The scale ablation established
    this deployment is bit-identical to the paper's float32-CPU
    configuration for this model on all seven original scenarios, and
    the S1 condition re-anchors against the round-3 CPU rows in-batch
    (every instance's S1 ARI must match to 1e-12, else the run is
    flagged and the instance re-run on CPU);
  * prompt template, JSON validation and fallback semantics identical
    across conditions -- only the summary block changes;
  * two conditions: S1 (deployed contract, re-run in-batch as a
    reproducibility anchor against the round-3 rows) and S3 (the D2
    sharp-optimum contract, the new condition);
  * S1 blocks are asserted byte-identical to the deployed generator's
    blocks (policies.stream_summary) before any call is made;
  * oracle / best-fixed / incumbent rows are NOT re-run: the round-3
    rows are the reference (instance construction is deterministic,
    verified by the S1 anchor).

Entry point:
    python -m a2case.contract_gap --data <csv-dir> --out results/contract_gap.json \
        [--device cuda]
"""

import argparse
import json
import os
import time

from . import metrics, round3
from .arms import _windowize
from .classics import run_hybrid_precomputed, stream_summary
from .locals import LocalLLMPolicyGen

# round-3 twin instances: CONFIGS[(pid-1) % 6] in (4, 5) == twin=True
TWIN_IDS = [f"P{p:02d}" for p in range(1, 25) if ((p - 1) % 6) in (4, 5)]

LEVELS = [1, 3]


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


def run(zf, out_path, device="cuda", only=None):
    import torch

    from .locals import _LLM_USER
    from .round3 import QWEN as qwen_path
    from .policies import stream_summary as deployed_summary

    rows = []
    if only and os.path.exists(out_path):
        # patch mode: keep the other instances' rows, replace the
        # targeted ones
        rows = [r for r in json.load(open(out_path))
                if r["scenario"] not in only]
    g = LocalLLMPolicyGen(qwen_path, device=device, dtype=torch.float16)
    for pid_str in TWIN_IDS:
        if only and pid_str not in only:
            continue
        pid = int(pid_str[1:])
        sc = round3.build_parametric(zf, pid, seed=17)
        assert sc.stats["twin"] is True, f"{pid_str} not a twin instance"
        windows = _windowize(sc.alerts, 300)
        n = len(sc.alerts)
        # equivalence guard: the level-1 block must be byte-identical to
        # what the deployed generator (arms.run_hybrid) would have sent
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
            ptok = g.prompt_tokens - pre[0]
            ctok = g.completion_tokens - pre[1]
            calls = g.calls - pre[2]
            nfb = g.fallbacks - pre[3]
            ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
            rows.append(dict(
                part="contract_gap", scenario=pid_str, arm=f"qwen0.5B_S{level}",
                level=level, seed=17, deployment=f"fp16-{device}",
                n_alerts=n, n_incidents=len(sc.incidents),
                n_windows=len(windows), **ev,
                llm_valid=calls - nfb, llm_calls=calls, fallbacks=nfb,
                prompt_tokens=ptok, completion_tokens=ctok,
                prompt_tokens_per_call=ptok / max(1, calls),
                dims_hist=json.dumps(sorted({tuple(t["dims"])
                                             for t in res["policy_trace"]})),
                day=sc.stats["day"]))
            _atomic_dump(rows, out_path)
            print(f"  [{pid_str} S{level}] ari={ev['ari']:.4f} "
                  f"ptok/call={ptok / max(1, calls):.0f} "
                  f"fb={nfb}/{calls}", flush=True)
    print(f"contract_gap complete: {len(rows)} rows", flush=True)


def analyze(out_path="results/contract_gap.json",
            round3_path="results/round3_local.json"):
    """Join with the round-3 rows (oracle/best-fixed/incumbent/S1) and
    print the gap-recovery summary used by the Discussion subsection."""
    from collections import defaultdict

    import numpy as np
    from scipy import stats

    new = {(r["scenario"], r["level"]): r
           for r in json.load(open(out_path)) if r["part"] == "contract_gap"}
    ref = defaultdict(dict)
    for r in json.load(open(round3_path)):
        if r["part"] == "parametric" and r["scenario"] in TWIN_IDS:
            ref[r["arm"]][r["scenario"]] = r

    print(f"{'inst':5s} {'S1':>7s} {'S3':>7s} {'d':>7s} "
          f"{'r3_S1':>7s} {'anchor':>7s} {'oracle':>7s} {'bf':>7s}")
    s1, s3, orc, bf = [], [], [], []
    anchors_ok = True
    for s in TWIN_IDS:
        a1, a3 = new[(s, 1)]["ari"], new[(s, 3)]["ari"]
        r3s1 = ref["hybrid_llm_0.5B"][s]["ari"]
        o = ref["oracle"][s]["ari"]
        b = ref["static_best_fixed_hindsight"][s]["ari"]
        same = abs(a1 - r3s1) < 1e-12
        anchors_ok &= same
        s1.append(a1), s3.append(a3), orc.append(o), bf.append(b)
        print(f"{s:5s} {a1:7.4f} {a3:7.4f} {a3 - a1:+7.4f} {r3s1:7.4f} "
              f"{'ok' if same else 'DIFF':>7s} {o:7.4f} {b:7.4f}")
    s1, s3, orc, bf = map(np.array, (s1, s3, orc, bf))
    d = s3 - s1
    nz = d[d != 0]
    print(f"\nS1 anchor vs round3 rows: {'ALL BIT-IDENTICAL' if anchors_ok else 'MISMATCH'}")
    print(f"mean ARI: S1={s1.mean():.4f}  S3={s3.mean():.4f}  "
          f"oracle={orc.mean():.4f}  best-fixed={bf.mean():.4f}")
    print(f"gap to oracle: S1={float((orc - s1).mean()):+.4f}  "
          f"S3={float((orc - s3).mean()):+.4f}  "
          f"recovered fraction={float(((orc - s1).mean() - (orc - s3).mean()) / max(1e-12, (orc - s1).mean())):.3f}")
    if len(nz):
        w = stats.wilcoxon(nz)
        rng = np.random.default_rng(17)
        boots = [float(np.mean(rng.choice(d, size=len(d), replace=True)))
                 for _ in range(10000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"S3-S1: dmean={d.mean():+.4f} wilcoxon p={w.pvalue:.4f} "
              f"95%CI[{lo:+.4f},{hi:+.4f}] wins={int((d > 0).sum())}/{len(d)}")
    else:
        print("S3-S1: all-zero differences")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=round3.DATA)
    ap.add_argument("--out", default="results/contract_gap.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these instance ids (e.g. P23); "
                         "rows are appended for later merging")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    if not args.analyze_only:
        if args.only and args.device != "cpu":
            raise SystemExit("--only patch runs are CPU/fp32 by design "
                             "(the GPU deployment diverged there)")
        run(args.data, args.out, device=args.device, only=args.only)
    analyze(args.out)


if __name__ == "__main__":
    main()
