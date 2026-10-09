"""Mapping-noise sensitivity sweep (stage 2; advisor next_step.md #2).

Question: are the arm-level ARI differences robust to the reference
partition's mapping noise at the rate the manual audit observes? All
grouping arms are label-agnostic, so perturbing ground truth requires no
re-grouping: rebuild each instance's stream (same seeds -> identical
partitions), perturb the per-alert incident assignment, re-score.

Perturbation model (fixed before the audit result is known): each attack
alert is reassigned, independently with probability e, to a uniformly
random *other* gt incident of the same scenario instance. e-grid
{0.0, 0.02, 0.05, 0.10, 0.15}: 0.0 must reproduce the round-3 ARI values
exactly (identity check); the audit's observed precision picks the
relevant e = 1 - precision for the paper's report; the full grid shows
the robustness band.

Arms scored: the five arms the subgroup analysis rests on (incumbent,
hindsight-best fixed, PSH, 0.5B LLM, oracle). The 0.5B arm is optional
(--llm); without it the sweep is CPU-cheap apart from the oracle.

Entry point:
    python -m a2c.audit_sensitivity --data <csv-dir> \
        --out results/audit_sensitivity.json [--llm] \
        [--instances P01,P02,... | --instances all]
"""

import argparse
import json
import os
import random
import tempfile
import time

import numpy as np

from . import metrics, round3
from .round3 import MINILM, QWEN, build_parametric, free_arms
from .locals import LocalLLMPolicyGen, MiniLMEmbedder

E_GRID = [0.0, 0.02, 0.05, 0.10, 0.15]
PERTURB_SEED = 20260922
REPORT_ARMS = ("static_incumbent", "static_best_fixed_hindsight",
               "hybrid_psh", "hybrid_llm_0.5B", "oracle")


def save_json(path, obj, retries=4):
    """Atomic write with retries: the workspace is sync-client backed and
    transiently locks result files."""
    for k in range(retries):
        try:
            d = os.path.dirname(path) or "."
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=1)
            os.replace(tmp, path)
            return
        except PermissionError:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def perturb(alerts, e, rng):
    """Independent reassignment of attack alerts to a random other
    incident; benign alerts untouched. Mutates gt_incident in place."""
    incidents = sorted({a.gt_incident for a in alerts if a.is_attack})
    if len(incidents) < 2:
        return 0
    n = 0
    for a in alerts:
        if not a.is_attack or rng.random() >= e:
            continue
        others = [i for i in incidents if i != a.gt_incident]
        a.gt_incident = rng.choice(others)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/audit_sensitivity.json")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--instances", default="all",
                    help="'all' (P01-P24) or comma-separated subset")
    args = ap.parse_args()
    zf = args.data

    if args.instances == "all":
        pids = list(range(1, 25))
    else:
        pids = [int(s[1:]) for s in args.instances.split(",")]

    rows = []
    minilm = MiniLMEmbedder(MINILM)
    llm = None
    for pid in pids:
        sc = build_parametric(zf, pid, seed=17)
        # arms are label-agnostic: group once, re-score under each e
        d = free_arms(sc, minilm)
        if args.llm:
            if llm is None:
                llm = LocalLLMPolicyGen(QWEN)
            d["hybrid_llm_0.5B"] = arms_run(sc, llm)
        arms_here = [a for a in REPORT_ARMS if d.get(a) is not None]
        orig = {id(a): a.gt_incident for a in sc.alerts if a.is_attack}
        n = len(sc.alerts)
        for e in E_GRID:
            # restore pristine labels, then apply this rate's perturbation
            for a in sc.alerts:
                if a.is_attack:
                    a.gt_incident = orig[id(a)]
            # deterministic per (e, instance): documented integer mix
            rng = random.Random(PERTURB_SEED * 100000
                                + round(e * 10000) * 100 + pid)
            n_swapped = perturb(sc.alerts, e, rng)
            for arm in arms_here:
                ev = metrics.evaluate(d[arm]["cases"], n, len(sc.incidents))
                rows.append(dict(
                    part="audit_sensitivity", e=e, scenario=sc.name,
                    arm=arm, seed=17, perturb_seed=PERTURB_SEED,
                    n_swapped=n_swapped,
                    n_attack_alerts=sum(1 for a in sc.alerts if a.is_attack),
                    n_incidents=len(sc.incidents),
                    ari=ev["ari"], purity=ev["purity"],
                    under=ev["under"], over=ev["over"]))
            save_json(args.out, rows)
            print(f"{sc.name} e={e}: swapped={n_swapped} "
                  + " ".join(f"{a}={metrics.evaluate(d[a]['cases'], n, len(sc.incidents))['ari']:.3f}"
                             for a in arms_here), flush=True)
    print("sensitivity sweep complete:", len(rows), "rows", flush=True)


def arms_run(sc, g):
    from . import arms
    res = arms.run_hybrid(sc.alerts, generator=g)
    res["prompt_tokens"] = g.prompt_tokens
    res["completion_tokens"] = g.completion_tokens
    return res


if __name__ == "__main__":
    main()
