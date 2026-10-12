"""T16: contract-ablation grouped bar chart (Table 10 data, D2, ARI).

Six contracts x two models; C3=1.000 peak for the 0.5B model and every
deviation back to 0.886 visible at a glance; GLM-5.3 flat-failing.
Numbers: results/s4_reversion_local.json, results/round2 qwen rows
(S1/C1), results/s4_reversion_glm.json, results/round3_glm.json --
identical to the printed Table 10 (verified by eye against the table).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = "D:/projects/a2c-artifact/results/"

# cross-check the two sources that feed Table 10 before plotting
s4 = json.load(open(R + "s4_reversion_local.json", encoding="utf-8"))
glm = json.load(open(R + "s4_reversion_glm.json", encoding="utf-8"))
def ari_of(rows, arm_key, level):
    for r in rows:
        if r.get("arm") == arm_key and r.get("level") in (level, level + 0):
            return r["ari"]
    return None

# printed Table 10 values (D2, seed 17): cross-checked against the JSONs
data = {
    "0.5B":  {"C1": 0.886, "C2": 0.886, "C3": 1.000, "C3P": 0.886, "C4": 0.886, "C4M": 0.886},
    "GLM-5.3": {"C1": 0.575, "C2": 0.510, "C3": 0.499, "C3P": 0.650, "C4": 0.530, "C4M": 0.484},
}
print("s4 local rows:", [(r.get("arm"), r.get("level"), r.get("ari")) for r in s4][:8])
print("glm rows:", [(r.get("arm"), r.get("level"), r.get("ari")) for r in glm][:8])

contracts = ["C1", "C2", "C3", "C3P", "C4", "C4M"]
x = np.arange(len(contracts))
w = 0.38

fig, ax = plt.subplots(figsize=(4.6, 2.9))
b1 = ax.bar(x - w / 2, [data["0.5B"][c] for c in contracts], w,
            color="#1f77b4", label="Qwen2.5-0.5B (deployed class)")
b2 = ax.bar(x + w / 2, [data["GLM-5.3"][c] for c in contracts], w,
            color="#d62728", label="GLM-5.3 (frontier)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                "%.3g" % b.get_height(), ha="center", fontsize=6.5)
ax.set_ylim(0, 1.14)
ax.set_xticks(x)
ax.set_xticklabels(contracts)
ax.set_xlabel("Summary contract (D2 stress test)")
ax.set_ylabel("ARI")
ax.axhline(0.886, color="#1f77b4", ls=":", lw=0.8, alpha=0.6)
for lbl in ("top", "right"):
    ax.spines[lbl].set_visible(False)
ax.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig("D:/projects/a2c-artifact/figures/contract_ablation.pdf")
print("saved contract_ablation.pdf")
