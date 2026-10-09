import os
"""Formatter smoke test for s4_reversion (no LLM calls).

Checks, on the real D2 build: (1) the S4M matrix text carries exactly the
same field values as the S4 dict (information equivalence); (2) S3P adds
only family_attacker_counts on top of S3 and has no cross-tab; (3) summary
block token counts per condition (the Hypothesis-A mechanism statistic).
"""
import json
import sys

from . import scenarios
from .arms import _windowize
from .s4_reversion import CONDITIONS, measure_summary_tokens, s3plus_summary, \
    s4_matrix_text

DATA = os.environ.get("A2C_DATA_DIR") or os.path.expanduser(
    os.path.join("~", ".cache", "kagglehub", "datasets", "chethuhn",
    "network-intrusion-dataset", "versions", "1"))
QWEN = os.environ.get("A2C_QWEN_DIR", r"..\data\models\Qwen2.5-0.5B-Instruct")


def main():
    sc = scenarios.build_twins(DATA, seed=17)
    print("D2 stats:", json.dumps(
        {k: sc.stats[k] for k in ("alerts_total", "incidents", "twin_alerts",
                                  "construction")}))
    windows = _windowize(sc.alerts, 300)
    print("windows:", len(windows))

    # pick the densest window (the mixed regime S4 reversion lives in)
    w = max(windows, key=len)
    s4 = CONDITIONS[1][1](w)
    txt = s4_matrix_text(w)
    s3p = s3plus_summary(w)
    s3 = CONDITIONS[0][1](w)

    # 1) information equivalence S4 dict <-> S4M text
    assert txt.startswith(f"n={s4['n']}")
    for f, n in s4["families"]:
        assert f"{f}={n}" in txt, (f, n)
    for d, n in s4["top_dsts"]:
        assert f"{d}={n}" in txt
    for p, n in s4["top_pairs"]:
        assert f"{p}={n}" in txt
    for p, n in s4["top_srcs"]:
        assert f"{p}={n}" in txt
    for fam, srcs in s4["family_sources"].items():
        for p, n in srcs.items():
            assert f"{p}={n}" in txt
    for fam, dst, src, n in s4["family_dst_src"]:
        assert f"{fam}|{dst}|{src}={n}" in txt
    print("S4 <-> S4M information equivalence: OK "
          f"({len(s4['family_dst_src'])} cross entries checked)")

    # 2) S3P = S3 + attacker counts only. Counts must equal the true
    # distinct-source cardinality per family (which family_sources'
    # most_common(4) truncation can understate -- that understatement is
    # exactly the new information S3P adds).
    assert set(s3p) - set(s3) == {"family_attacker_counts"}
    assert "family_dst_src" not in s3p
    truth = {}
    for a in w:
        truth.setdefault(a.family, set()).add(a.src)
    for f, k in s3p["family_attacker_counts"].items():
        assert k == len(truth[f]), (f, k, len(truth[f]))
    understated = [f for f, k in s3p["family_attacker_counts"].items()
                   if len(s3p["family_sources"][f]) < k]
    print("S3P = S3 + family_attacker_counts only: OK "
          f"(e.g. {dict(list(s3p['family_attacker_counts'].items())[:3])}; "
          f"truncation understates {len(understated)} families)")

    # 3) token counts (deployed 0.5B tokenizer)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    stats = measure_summary_tokens(windows, tok, "local")
    for name, st in stats.items():
        print(f"tokens[{name}]: mean={st['mean']:.0f} p50={st['p50']} "
              f"p90={st['p90']} max={st['max']}")
    ratio = stats["S4M"]["mean"] / stats["S4"]["mean"]
    print(f"S4M/S4 mean token ratio: {ratio:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
