"""One-off repair: recompute the per-window twin-pair counts stored in
results/contract_gap_n20.json policy traces.

The prereg's twin-dominated rule counts "both campaigns of the twin pair"
per window. The original recorder matched a.family against the dataset
label (e.g. 'FTP-Patator'), but parametric alerts carry the signature
catalog's classification families (e.g. 'BRUTE-FTP'), so every count
stored during the run is 0. Windows are a deterministic function of the
stream, so the counts are recomputed by rebuilding each instance and
matching gt_incident against the twin pair ({i, i+'-twin'} for the
instance's twin incidents) --- no model outputs are touched; ARI and all
per-window policy choices are unchanged. Run AFTER the GPU job and any
--only patches have finished writing the file.

Entry (from experiments/):
    python -m a2c.trace_patch
"""

import json

from . import round3, scenarios
from .arms import _windowize
from .contract_gap_n20 import _atomic_dump, _build


def main(path="results/contract_gap_n20.json"):
    rows = json.load(open(path))
    fixed = 0
    for r in rows:
        if "policy_trace" not in r or not r.get("policy_trace"):
            continue
        tag = r["scenario"]
        base = tag.split("-s")[0]
        if base == "D2":
            sc = scenarios.build_twins(round3.DATA, seed=r["seed"])
        else:
            sc = _build(round3.DATA, int(base[1:]), seed=r["seed"])
        twins = {str(i) for i in sc.incidents if str(i).endswith("-twin")}
        pair = twins | {i[:-5] for i in twins}
        windows = _windowize(sc.alerts, 300)
        assert len(windows) == len(r["policy_trace"]), \
            f"{tag}: window count mismatch {len(windows)} vs trace"
        for t in r["policy_trace"]:
            w = windows[t["window"]]
            t["n_twin_src"] = sum(1 for a in w if a.src == round3.TWIN_SRC)
            t["n_twin_family"] = sum(
                1 for a in w if a.is_attack and str(a.gt_incident) in pair)
            fixed += 1
    _atomic_dump(rows, path)
    print(f"trace_patch: recomputed twin counts on {fixed} traces "
          f"({len(rows)} rows) -> {path}")


if __name__ == "__main__":
    main()
