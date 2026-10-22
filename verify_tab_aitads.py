"""Verify tab:aitads (paper's External Validity table) against stored results.

Checks, per scenario (7 informative + shaw shown for completeness):
  - incumbent (src,dst), sid, family,dst, family, embedding(w2v,tau=.75),
    PSH, LLM 0.5B per-scenario ARI (3 decimals, as printed in the paper)
  - the caption's three paired tests vs the exact recomputation:
    LLM - incumbent   mean +0.052, 6 pos / 1 neg, exact Wilcoxon p = 0.031
    LLM - sid         mean -0.156, 0/7,           exact Wilcoxon p = 0.016
    embedding - inc.  mean +0.391, 7/0,           exact Wilcoxon p = 0.016
Run from the repo root:  python verify_tab_aitads.py
"""
import glob
import json
import os

RES = os.path.join(os.path.dirname(__file__), "results", "gate0")
SCEN = ["fox", "harrison", "russellmitchell", "santos", "wardbeck",
        "wheeler", "wilson"]          # informative seven (shaw excluded)
DEC = 3

# tab:aitads columns as printed in the paper
TEX = {  # scen: (srcdst, sid, familydst, family, emb, psh, llm)
    "fox":            (0.016, 0.527, 0.162, 0.948, 0.294, 0.016, 0.162),
    "harrison":       (0.036, 0.183, 0.098, 0.277, 0.277, 0.024, 0.098),
    "russellmitchell":(0.137, 0.412, 0.148, 0.948, 0.947, 0.032, 0.148),
    "santos":         (0.031, 0.330, 0.112, 0.555, 0.626, 0.019, 0.112),
    "shaw":           (0.000, 0.000, 0.000, 1.000, 1.000, 0.000, 0.000),
    "wardbeck":       (0.010, 0.186, 0.041, 0.381, 0.469, 0.010, 0.041),
    "wheeler":        (0.191, 0.238, 0.225, 0.256, 0.256, 0.198, 0.225),
    "wilson":         (0.427, 0.428, 0.427, 0.718, 0.718, 0.113, 0.427),
}

matrix = {}
for r in json.load(open(os.path.join(RES, "aitads_matrix.json")))["rows"]:
    if r["arm"] == "static":
        matrix[(r["scenario"], r["dimset"])] = r["ari"]
    elif r["arm"] == "embedding":
        matrix[(r["scenario"], "emb")] = r["ari"]
    elif r["arm"] == "hybrid-PSH":
        matrix[(r["scenario"], "psh")] = r["ari"]
llm = {}
for f in glob.glob(os.path.join(RES, "llm05", "*.json")):
    llm[os.path.basename(f)[:-5]] = json.load(open(f))["ari"]

problems = []
for scen, texrow in TEX.items():
    got = (round(matrix[(scen, "src,dst")], DEC),
           round(matrix[(scen, "sid")], DEC),
           round(matrix[(scen, "family,dst")], DEC),
           round(matrix[(scen, "family")], DEC),
           round(matrix[(scen, "emb")], DEC),
           round(matrix[(scen, "psh")], DEC),
           round(llm[scen], DEC))
    if got != texrow:
        problems.append(f"{scen}: paper {texrow} != stored {got}")

# paired tests, exact recomputation (same conventions as a2case.aitads_compare)
from scipy import stats as st

def paired(a_key, b_key):
    d = []
    for s in SCEN:
        a = llm[s] if a_key == "llm" else matrix[(s, a_key)]
        b = llm[s] if b_key == "llm" else matrix[(s, b_key)]
        d.append(a - b)
    pos = sum(x > 0 for x in d); neg = sum(x < 0 for x in d)
    return sum(d) / len(d), pos, neg, st.wilcoxon(d, alternative="two-sided",
                                                  mode="exact").pvalue

TEX_TESTS = {  # (mean, pos, neg, p) as stated in the paper caption/registry
    ("llm", "src,dst"): (0.052, 6, 1, 0.031),
    ("llm", "sid"):     (-0.156, 0, 7, 0.016),
    ("emb", "src,dst"): (0.391, 7, 0, 0.016),
}
for (a, b), (tm, tp, tn, tpv) in TEX_TESTS.items():
    m, p, n, pv = paired(a, b)
    if abs(m - tm) > 5e-4 or (p, n) != (tp, tn) or abs(pv - tpv) > 5e-4:
        problems.append(f"test {a}-{b}: paper ({tm},{tp}/{tn},p={tpv}) "
                        f"!= recomputed ({m:.4f},{p}/{n},p={pv:.4f})")

if problems:
    print("FAIL")
    for x in problems:
        print(" -", x)
    raise SystemExit(1)
print("verify_tab_aitads: OK (8 scenarios x 7 columns + 3 paired tests)")
