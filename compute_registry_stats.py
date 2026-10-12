"""T7 registry effect sizes, take 2.

mixed-20 LLM column = pre-registered contract_gap_n20 S1 (P23 CPU row per protocol),
other arms from round3_local + round3_expansion (matches incumbent_sensitivity log).
"""
import json
import numpy as np
from scipy import stats

R = "D:/projects/a2c-artifact/results/"
load = lambda n: json.load(open(R + n, encoding="utf-8"))

r3 = load("round3_local.json")
exp = load("round3_expansion.json")
cg = load("contract_gap_n20.json")

g = {}
for r in r3 + exp:
    if r.get("seed", 17) != 17:
        continue
    g.setdefault(r["arm"], {})[r["scenario"]] = r["ari"]

# S1 values on the 20 mixed instances; protocol: P23 takes the fp32-cpu row
s1 = {}
for r in cg:
    if r["arm"] != "qwen0.5B_S1" or not r["scenario"].startswith("P"):
        continue
    s = r["scenario"]
    if s == "P23":
        if r.get("deployment") == "fp32-cpu":
            s1[s] = r["ari"]
        continue
    if s.endswith(("-s29", "-s43")):
        continue
    s1[s] = r["ari"]
assert len(s1) == 20, len(s1)

MIXED8 = ["P%02d" % p for p in range(1, 25) if (p - 1) % 6 in (4, 5)]
MIXED20 = MIXED8 + ["P%02d" % p for p in range(25, 37)]
SINGLE16 = ["P%02d" % p for p in range(1, 25) if (p - 1) % 6 not in (4, 5)]
ALL24 = ["P%02d" % p for p in range(1, 25)]

def armv(arm, scens):
    return np.array([g[arm][s] for s in scens])

def llmv(scens):
    # mixed-20 uses the pre-registered S1 batch; single-16 uses the deployed round3_local arm
    return np.array([s1[s] if s in s1 else g["hybrid_llm_0.5B"][s] for s in scens])

def wil(d):
    d = d[d != 0]
    res = stats.wilcoxon(d)
    ranks = stats.rankdata(np.abs(d))
    rrb = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()
    return res.pvalue, rrb, len(d)

def cliffs(d1, d2):
    return ((d1[:, None] > d2[None, :]).sum() - (d1[:, None] < d2[None, :]).sum()) / (len(d1) * len(d2))

inc, bf, orc, psh = "static_incumbent", "static_best_fixed_hindsight", "oracle", "hybrid_psh"

print("=== mixed-20 column check vs tab:subgroup ===")
for nm, v in [("inc", armv(inc, MIXED20)), ("bf", armv(bf, MIXED20)),
              ("oracle", armv(orc, MIXED20)), ("LLM S1", llmv(MIXED20)),
              ("psh", armv(psh, MIXED20))]:
    print(f"  {nm:8s} mean={v.mean():.4f}")
print("  paper: inc 0.713 bf 0.963 oracle 0.996 LLM 0.693 psh 0.679")

print()
print("=== subgroup tests (as published) ===")
d_li_s = llmv(SINGLE16) - armv(inc, SINGLE16)
d_li_m = llmv(MIXED20) - armv(inc, MIXED20)
d_bl_s = armv(bf, SINGLE16) - llmv(SINGLE16)
d_bl_m = armv(bf, MIXED20) - llmv(MIXED20)
d_ob_s = armv(orc, SINGLE16) - armv(bf, SINGLE16)
d_ob_m = armv(orc, MIXED20) - armv(bf, MIXED20)
d_ol_s = armv(orc, SINGLE16) - llmv(SINGLE16)
d_ol_m = armv(orc, MIXED20) - llmv(MIXED20)

for nm, ds, dm in [("LLM-inc", d_li_s, d_li_m), ("BF-LLM", d_bl_s, d_bl_m),
                   ("ORC-BF", d_ob_s, d_ob_m), ("ORC-LLM", d_ol_s, d_ol_m)]:
    ps_, rs_, ns_ = wil(ds)
    pm_, rm_, nm_ = wil(dm)
    u, pi = stats.mannwhitneyu(ds, dm, alternative="two-sided")
    print(f"{nm}: single {ds.mean():+.3f} p={ps_:.4g} r={rs_:+.3f} | mixed {dm.mean():+.3f} p={pm_:.4g} r={rm_:+.3f} | MW p={pi:.4g} delta={cliffs(ds, dm):+.3f}")
print("paper: LLM-inc single +0.361 p=0.002 | mixed -0.020 p=0.55 | inter 0.002")
print("paper: BF-LLM single +0.007 p=0.25 | mixed +0.270(=-LLM-BF -0.270) p=0.0001 | inter 2.3e-7")
print("paper: ORC-BF single -0.006 p=0.16 | mixed +0.033 p=0.38")
print("paper: ORC-LLM single +0.001 p=1.00 | mixed +0.303 p<0.001")

print()
print("=== contract contrasts ===")
lev = {}
for r in cg:
    if not r["scenario"].startswith("P"):
        continue
    lev.setdefault((r["arm"], r["scenario"]), {})[r["deployment"]] = r["ari"]
s3 = {}
for r in cg:
    if r["arm"] != "qwen0.5B_S3" or not r["scenario"].startswith("P"):
        continue
    s = r["scenario"]
    if s == "P23":
        if r.get("deployment") == "fp32-cpu":
            s3[s] = r["ari"]
        continue
    if s.endswith(("-s29", "-s43")):
        continue
    s3[s] = r["ari"]
assert len(s3) == 20
d_s3s1 = np.array([s3[s] - s1[s] for s in MIXED20])
p_c, r_c, n_c = wil(d_s3s1)
print(f"n=20 S3-S1: mean={d_s3s1.mean():+.4f} p={p_c:.4g} r={r_c:+.3f} W/T/L={int((d_s3s1>0).sum())}/{int((d_s3s1==0).sum())}/{int((d_s3s1<0).sum())}")
d_os3 = armv(orc, MIXED20) - np.array([s3[s] for s in MIXED20])
p_d, r_d, n_d = wil(d_os3)
print(f"n=20 ORC-S3: mean={d_os3.mean():+.4f} p={p_d:.4g} r={r_d:+.3f}")
print("paper: +0.135 p=0.048 11W/9L; oracle-S3 +0.168 p=6e-5")

cg8 = load("contract_gap.json")
l8 = {}
for r in cg8:
    l8.setdefault((r["arm"], r["level"]), {})[r["scenario"]] = r["ari"]
sc8 = sorted(set(r["scenario"] for r in cg8))
d8 = np.array([l8[("qwen0.5B_S3", 3)][s] - l8[("qwen0.5B_S1", 1)][s] for s in sc8])
p_e, r_e, n_e = wil(d8)
print(f"n=8 scens={sc8}")
print(f"n=8 S3-S1: mean={d8.mean():+.4f} p={p_e:.4g} r={r_e:+.3f}")
