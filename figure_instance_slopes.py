"""T15: paired slope chart, 24 parametric instances, incumbent vs LLM 0.5B.

single-regime (16) vs mixed-regime (8) in two colors; oracle per instance
as the faint reference line. Data: round3_local.json seed-17 rows (the
same source behind tab:stats/subgroup single-16 column).
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "D:/projects/a2c-artifact/results/"
rows = json.load(open(R + "round3_local.json", encoding="utf-8"))
g = {}
for r in rows:
    if r.get("seed", 17) != 17 or r["part"] != "parametric":
        continue
    g.setdefault(r["arm"], {})[r["scenario"]] = r["ari"]

MIXED = ["P%02d" % p for p in range(1, 25) if (p - 1) % 6 in (4, 5)]
SINGLE = ["P%02d" % p for p in range(1, 25) if (p - 1) % 6 not in (4, 5)]
inc, llm, orc = "static_incumbent", "hybrid_llm_0.5B", "oracle"

# sort instances by incumbent ARI so the left column reads as a ranking
order = sorted(SINGLE + MIXED, key=lambda s: g[inc][s])

fig, ax = plt.subplots(figsize=(4.6, 3.4))
C_SINGLE = "#1f77b4"
C_MIXED = "#d62728"
C_ORACLE = "#999999"

for s in order:
    grp = MIXED if s in MIXED else SINGLE
    color = C_MIXED if grp == MIXED else C_SINGLE
    alpha = 0.85 if grp == MIXED else 0.45
    lw = 1.4 if grp == MIXED else 1.0
    y0, y1 = g[inc][s], g[llm][s]
    ax.plot([0, 1], [y0, y1], color=color, alpha=alpha, lw=lw,
            marker="o", ms=2.6, zorder=3 if grp == MIXED else 2)
    # oracle reference for the same instance
    ax.plot([0, 1], [g[orc][s], g[orc][s + "_orc"] if False else g[orc][s]],
            color=C_ORACLE, alpha=0.35, lw=0.7, zorder=1)

ax.set_xlim(-0.12, 1.12)
ax.set_ylim(-0.03, 1.05)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Incumbent (deployed)", "LLM 0.5B (adaptive)"])
ax.set_ylabel("ARI (per instance)")
for lbl in ("top", "right"):
    ax.spines[lbl].set_visible(False)

# regime means as side annotation
import numpy as np
ms = (np.mean([g[llm][s] - g[inc][s] for s in SINGLE]),
      np.mean([g[llm][s] - g[inc][s] for s in MIXED]))
bbox = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)
ax.text(0.02, 0.015, "single-regime mean $\\Delta$ = %+.2f (16 instances)" % ms[0],
        color=C_SINGLE, fontsize=8, transform=ax.transAxes, bbox=bbox)
ax.text(0.02, 0.115, "mixed-regime mean $\\Delta$ = %+.2f (8 instances)" % ms[1],
        color=C_MIXED, fontsize=8, transform=ax.transAxes, bbox=bbox)
ax.text(0.98, 0.965, "gray: oracle (per instance)", color=C_ORACLE,
        fontsize=8, ha="right", transform=ax.transAxes)

fig.tight_layout()
fig.savefig("D:/projects/a2c-artifact/figures/instance_slopes.pdf")
print("saved instance_slopes.pdf; means:", [round(float(m), 3) for m in ms])
