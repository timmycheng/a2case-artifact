"""Generate paper figures + LaTeX tables from results.json."""

import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 9, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})

ARM_LABEL = {"static": "Static rules", "embedding": "Embedding clustering",
             "hybrid_psh": "Generated policy (PSH)", "triage_ref": "Per-alert triage (ref.)"}
COLOR = {"static": "#888888", "embedding": "#4C72B0", "hybrid_psh": "#DD8452"}


def load(results_path):
    with open(results_path) as f:
        return json.load(f)["results"]


def mean_by(results, filt, key="ari"):
    acc = defaultdict(list)
    for r in results:
        if filt(r):
            acc[r["arm"]].append(r[key])
    return {a: float(np.mean(v)) for a, v in acc.items()}


def fig_scenario_bars(results, out):
    names = ["A1", "A2", "A3", "E"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    metric_list = [("ari", "Adjusted Rand index", False),
                   ("purity", "Case purity", False),
                   ("compression", "Alerts per case", True)]
    for ax, (m, title, logy) in zip(axes, metric_list):
        width = 0.26
        xs = np.arange(len(names))
        for i, arm in enumerate(["static", "embedding", "hybrid_psh"]):
            vals = []
            for n in names:
                v = [r[m] for r in results
                     if r["scenario"] == n and r["arm"] == arm and r["seed"] == 17]
                vals.append(np.mean(v) if v else 0)
            ax.bar(xs + (i - 1) * width, vals, width, label=ARM_LABEL[arm],
                   color=COLOR[arm])
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        if logy:
            ax.set_yscale("log")
            ax.set_ylabel("alerts/case (log)", fontsize=8)
            ax.set_ylim(1, 3000)
        else:
            ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_cost_quality(results, out, llm_local=None, llm_glm=None):
    """Cost-quality plane: ARI vs measured LLM cost (USD per 10k alerts).

    Zero-LLM-cost arms are clamped to 1e-4 on the log axis. LLM 0.5B cost is
    the paper's measured call pattern (739 calls, ~606 input + ~18 output
    tokens/call at the tokenizer) priced at gpt-4o-mini list rates — the
    normalization of Table tab:cost; GLM rows carry measured token counts and
    use the same normalized basis (own-price inversion is tab:llm's column).
    """
    scen_set = ("A1", "A2", "A3", "B2", "B3", "E")
    series = [
        ("static", "Static rules", "#888888", "o"),
        ("embedding", "Embedding clustering", "#4C72B0", "o"),
        ("hybrid_psh", "Generated policy (PSH)", "#DD8452", "o"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for arm, label, color, marker in series:
        xs, ys = [], []
        for r in results:
            if r["arm"] == arm and r["scenario"] in scen_set \
                    and r["seed"] == 17:
                xs.append(max(r["usd_per_10k"], 1e-4))
                ys.append(r["ari"])
        ax.scatter(xs, ys, label=label, color=color, marker=marker, s=28)
    mini_in, mini_out = 0.15e-6, 0.60e-6
    if llm_local:
        xs, ys = [], []
        for r in llm_local:
            if r["scenario"] in scen_set:
                cost = r["llm_calls"] * (606 * mini_in + 18 * mini_out)
                xs.append(cost / r["n_alerts"] * 10000)
                ys.append(r["ari"])
        ax.scatter(xs, ys, label="Generated policy (LLM 0.5B)",
                   color="#55A868", marker="s", s=30)
    if llm_glm:
        xs, ys = [], []
        for r in llm_glm:
            if r["scenario"] in scen_set:
                cost = (r["prompt_tokens"] * mini_in
                        + r["completion_tokens"] * mini_out)
                xs.append(cost / r["n_alerts"] * 10000)
                ys.append(r["ari"])
        ax.scatter(xs, ys, label="Generated policy (GLM-5.3)",
                   color="#8172B3", marker="D", s=26)
    # triage reference anchor: analytic cost at 320 tokens/alert
    x_ref = 320 * (0.15 / 1e6) * 10_000
    ax.axvline(x_ref, color="#C44E52", ls="--", lw=1)
    ax.text(x_ref * 1.1, 0.02, "per-alert triage\nreference", fontsize=7,
            color="#C44E52")
    ax.set_xscale("log")
    ax.set_xlabel("LLM cost (USD per 10k alerts, log)")
    ax.set_ylabel("Adjusted Rand index")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_noise_sweep(results, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ratios = [1, 2, 4, 8, 16]
    for arm in ["static", "embedding", "hybrid_psh"]:
        ys = []
        for ratio in ratios:
            scen = "A2" if ratio == 1 else f"sweep-{ratio}"
            v = [r["ari"] for r in results
                 if r["arm"] == arm and r["scenario"] == scen and r["seed"] == 17]
            ys.append(np.mean(v) if v else np.nan)
        ax.plot(ratios, ys, "o-", label=ARM_LABEL[arm], color=COLOR[arm], ms=4)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ratios)
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_xlabel("False-positive : attack-alert ratio")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_window_ablation(results, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    deltas = [60, 300, 900, 1800]
    for arm in ["static", "hybrid_psh"]:
        ys = []
        for d in deltas:
            v = [r["ari"] for r in results
                 if r["arm"] == arm and r["scenario"] == f"win-{d}"]
            ys.append(np.mean(v) if v else np.nan)
        ax.plot([d // 60 for d in deltas], ys, "o-", label=ARM_LABEL[arm],
                color=COLOR[arm], ms=4)
    ax.set_xlabel("Micro-batch window $\\Delta$ (minutes)")
    ax.set_ylabel("Adjusted Rand index (Scenario E)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_tau_sweep(results, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    taus = [0.5, 0.6, 0.7, 0.8, 0.9]
    for scen, ls in [("E", "-"), ("A2", "--")]:
        ys = []
        for t in taus:
            v = [r["ari"] for r in results
                 if r["arm"] == "embedding" and r["scenario"] == f"tau-{t}"
                 and scmatch(r, scen)]
            ys.append(np.mean(v) if v else np.nan)
        ax.plot(taus, ys, ls, marker="o", ms=4, color=COLOR["embedding"],
                label=f"Embedding ({'E' if scen == 'E' else 'A2'})")
    ax.set_xlabel("Similarity threshold $\\tau$")
    ax.set_ylabel("Adjusted Rand index")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def scmatch(r, scen):
    return r["stats"].get("scenario") == scen if "stats" in r else False


def table_main(results, out_tex):
    rows = []
    for n in ["A1", "A2", "A3", "B1", "B2", "B3", "E"]:
        for arm in ["static", "embedding", "hybrid_psh"]:
            sel = [r for r in results
                   if r["scenario"] == n and r["arm"] == arm and r["seed"] == 17]
            if not sel:
                continue
            r = sel[0]
            rows.append(
                f"{n} & {ARM_LABEL[arm]} & {r['ari']:.2f} & {r['purity']:.2f} & "
                f"{r['under']:.2f} & {r['over']:.2f} & {r['compression']:.0f} & "
                f"{r['benign_contam']:.2f} & {r['usd_per_10k']:.2f} \\\\")
    with open(out_tex, "w") as f:
        f.write("\n".join(rows))


def main(results_path, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    results = load(results_path)
    res_dir = os.path.dirname(os.path.abspath(results_path))
    llm_local = llm_glm = None
    p_local = os.path.join(res_dir, "round3_local.json")
    p_glm = os.path.join(res_dir, "round3_glm.json")
    if os.path.exists(p_local):
        llm_local = [r for r in json.load(open(p_local, encoding="utf-8"))
                     if r.get("part") == "original_local"
                     and r["arm"] == "hybrid_llm_0.5B"]
    if os.path.exists(p_glm):
        llm_glm = [r for r in json.load(open(p_glm, encoding="utf-8"))
                   if r.get("part") == "original_glm"
                   and r["arm"] == "hybrid_llm_glm5.3"]
    fig_scenario_bars(results, os.path.join(fig_dir, "scenario_bars.pdf"))
    fig_cost_quality(results, os.path.join(fig_dir, "cost_quality.pdf"),
                     llm_local=llm_local, llm_glm=llm_glm)
    fig_noise_sweep(results, os.path.join(fig_dir, "noise_sweep.pdf"))
    fig_window_ablation(results, os.path.join(fig_dir, "window_ablation.pdf"))
    fig_tau_sweep(results, os.path.join(fig_dir, "tau_sweep.pdf"))
    table_main(results, os.path.join(fig_dir, "table_main.tex"))
    print("figures + tables written to", fig_dir)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2])


def fig_static_heatmap(sweep_rows, out):
    """ARI heatmap: dim-set x scenario (the 'no fixed winner' figure)."""
    dimsets = ["src,dst", "src,dst,port", "dst,port", "family",
               "family,dst", "sid", "src,family"]
    scens = ["A1", "A2", "A3", "B2", "B3", "E", "D2"]
    m = np.full((len(dimsets), len(scens)), np.nan)
    for r in sweep_rows:
        i, j = dimsets.index(r["config"]), scens.index(r["scenario"])
        m[i, j] = r["ari"]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    im = ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(scens)))
    ax.set_xticklabels(scens, fontsize=8)
    ax.set_yticks(range(len(dimsets)))
    ax.set_yticklabels(dimsets, fontsize=8)
    for i in range(len(dimsets)):
        for j in range(len(scens)):
            v = m[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if 0.25 < v < 0.9 else "white")
    ax.set_title("ARI of every fixed static configuration", fontsize=10,
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
