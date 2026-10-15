#!/usr/bin/env python
"""Regenerate the Figure 'window_sweep' (case-inactivity timeout sweep on the
replayed Thursday stream) from results/window_sweep.json.

Usage: python figure_window_sweep.py [results_dir] [out.pdf]
Values and styling follow the paper's Figure (sec:replay-results): incumbent
and timeout-restricted oracle ARI vs timeout w in {1,5,10,15,20,30} minutes,
with the campaign's largest quiet gap (24.6 min) marked as a vertical line.
"""
import json
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    out = sys.argv[2] if len(sys.argv) > 2 else "window_sweep.pdf"
    rows = json.load(open(os.path.join(results_dir, "window_sweep.json")))
    anatomy = next(r for r in rows if r["part"] == "burst_anatomy")
    sweep = sorted((r for r in rows if r["part"] == "window_sweep"),
                   key=lambda r: r["timeout_min"])
    xs = [r["timeout_min"] for r in sweep]
    inc = [r["incumbent"]["ari"] for r in sweep]
    ora = [r["oracle"]["ari"] for r in sweep]
    gap = anatomy["max_gap_min"]

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(xs, inc, "o-", ms=4, color="#1f77b4", label="static incumbent")
    ax.plot(xs, ora, "s--", ms=4, color="#d62728",
            label="oracle (timeout-restricted)")
    ax.axvline(gap, color="0.45", ls=":", lw=1.2,
               label=f"largest quiet gap ({gap:.1f} min)")
    ax.set_xlabel(r"case inactivity timeout $w$ (minutes)")
    ax.set_ylabel("ARI on R-Thursday")
    ax.set_ylim(-0.04, 1.05)
    ax.set_xticks(xs)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"window sweep figure written: {out} "
          f"(incumbent={inc}, oracle={ora}, gap={gap:.2f} min)")


if __name__ == "__main__":
    main()
