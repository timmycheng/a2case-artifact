"""Experiment runner: three arms x scenarios, ablations, cost accounting.

Outputs JSON results + CSV tables to experiments/results/.
Usage:  python -m a2case.run --data <dir-or-zip> --out results
"""

import argparse
import json
import os
import zipfile

from . import scenarios, arms, metrics

PRICE_TABLE = {
    # pinned list prices, USD per 1M tokens (as of Sept 2026, per vendor
    # price pages; used for the projection only, no calls are made)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-sonnet": (3.00, 15.00),
}
# per-alert triage reference: fixed prompt per alert (measured template)
PER_ALERT_PROMPT_TOKENS = 320


def run_all(data_path, out_dir, seeds=(17, 29, 43)):
    os.makedirs(out_dir, exist_ok=True)
    zf = zipfile.ZipFile(data_path) if data_path.endswith(".zip") else data_path
    results = []
    for seed in seeds:
        for name in [s[0] for s in scenarios.SCENARIO_INSTANCES]:
            sc = scenarios.build_scenario(zf, name, seed=seed)
            n = len(sc.alerts)
            print(f"[seed {seed}] {name}: {n} alerts, "
                  f"{len(sc.incidents)} incidents, ratio {sc.stats['noise_ratio']}")

            arms_out = {}
            arms_out["static"] = arms.run_static(sc.alerts)
            arms_out["embedding"] = arms.run_embedding(sc.alerts)
            arms_out["hybrid_psh"] = arms.run_hybrid(sc.alerts)

            for arm, res in arms_out.items():
                ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
                if arm == "hybrid_psh":
                    res["policy_calls"] = len(res["policy_trace"])
                cost = metrics.cost_row(res, n)
                if arm == "hybrid_psh":
                    # cost of the same architecture on an LLM generator:
                    # one call per window (measured), fixed prompt sizes
                    w = len(res["policy_trace"])
                    cost_llm = metrics.cost_row(
                        dict(res, prompt_tokens=w * 480,
                             completion_tokens=w * 60), n)
                    ev["usd_per_10k_llm_projected"] = cost_llm["usd_per_10k"]
                    cost["usd_per_10k"] = cost_llm["usd_per_10k"]
                    cost["policy_calls"] = w
                    cost["prompt_tokens"] = w * 480
                    cost["completion_tokens"] = w * 60
                results.append(dict(seed=seed, scenario=name, arm=arm,
                                    n_alerts=n, n_incidents=len(sc.incidents),
                                    **ev, **cost, stats=sc.stats))
            # per-alert triage reference: analytic cost anchor (no ARI)
            ref = metrics.cost_row({"prompt_tokens": 0, "completion_tokens": 0,
                                    "policy_calls": 1, "wall": 0.0}, n,
                                   est_in_tok_per_alert=PER_ALERT_PROMPT_TOKENS)
            results.append(dict(seed=seed, scenario=name, arm="triage_ref",
                                n_alerts=n, n_incidents=len(sc.incidents),
                                ari=None, purity=None, compression=1.0,
                                usd_per_10k=ref["usd_per_10k"],
                                usd_total=ref["usd_total"],
                                policy_calls=n, prompt_tokens=n * PER_ALERT_PROMPT_TOKENS,
                                completion_tokens=0, wall=0.0,
                                stats=sc.stats))
        # noise sweep on Wednesday (B2 gradient) for the noise-robustness curve
        for ratio in (2.0, 4.0, 16.0):
            from . import ingest as _ing
            streams = [_ing.build_stream(zf, "Wednesday-workingHours.pcap_ISCX.csv",
                                         "wednesday", ratio, seed)]
            alerts, incidents, stats = scenarios._merge(streams)
            n = len(alerts)
            for arm, res in [("static", arms.run_static(alerts)),
                             ("embedding", arms.run_embedding(alerts)),
                             ("hybrid_psh", arms.run_hybrid(alerts))]:
                ev = metrics.evaluate(res["cases"], n, len(incidents))
                cost = metrics.cost_row(res, n)
                results.append(dict(seed=seed, scenario=f"sweep-{ratio:g}", arm=arm,
                                    n_alerts=n, n_incidents=len(incidents),
                                    **ev, **cost, stats=stats))

    # window ablation (Scenario E, static + hybrid only; embedding fixed at 30m)
    scE = scenarios.build_scenario(zf, "E", seed=17)
    for delta in (60, 300, 900, 1800):
        for arm_name, res in [
            ("static", arms.run_static(scE.alerts, delta_sec=delta)),
            ("hybrid_psh", arms.run_hybrid(scE.alerts, delta_sec=delta)),
        ]:
            ev = metrics.evaluate(res["cases"], len(scE.alerts),
                                  len(scE.incidents))
            cost = metrics.cost_row(res, len(scE.alerts))
            results.append(dict(seed=17, scenario=f"win-{delta}", arm=arm_name,
                                n_alerts=len(scE.alerts),
                                n_incidents=len(scE.incidents),
                                **ev, **cost, stats=scE.stats))

    # embedding tau sweep on E and A2
    for scname in ("E", "A2"):
        sc = scenarios.build_scenario(zf, scname, seed=17)
        for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
            res = arms.run_embedding(sc.alerts, tau=tau)
            ev = metrics.evaluate(res["cases"], len(sc.alerts),
                                  len(sc.incidents))
            cost = metrics.cost_row(res, len(sc.alerts))
            results.append(dict(seed=17, scenario=f"tau-{tau}", arm="embedding",
                                n_alerts=len(sc.alerts),
                                n_incidents=len(sc.incidents),
                                **ev, **cost, stats=sc.stats))

    # budget upper bound (advisor request): worst-case full grid with frontier
    # model prices on hybrid-LLM projections + per-alert triage reference
    budget = budget_upper_bound(results)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(dict(results=results, budget=budget,
                       price_table=PRICE_TABLE,
                       per_alert_prompt_tokens=PER_ALERT_PROMPT_TOKENS), f, indent=1)
    print("budget:", json.dumps(budget, indent=1))
    return results


def budget_upper_bound(results):
    """Worst-case API budget for the full study (advisor request item 4).

    Conservative upper bound: every scenario instance x seed run with the
    hybrid arm on a frontier model (gpt-4o prices), x2 safety factor, plus
    one sampled per-alert triage reference run. Prices are USD per 1M
    tokens and are converted here; windows per run come from the measured
    policy-trace length (the LLM condition makes exactly one call per
    window).
    """
    p_in, p_out = PRICE_TABLE["gpt-4o"]
    p_in /= 1e6
    p_out /= 1e6
    hybrid_runs = [r for r in results if r["arm"] == "hybrid_psh"]
    runs_per_seed = len([r for r in hybrid_runs if r["seed"] == 17])
    windows_per_run = (sum(r["policy_calls"] + len(r.get("policy_trace", []))
                           for r in hybrid_runs) / max(len(hybrid_runs), 1))
    alerts_per_run = sum(r["n_alerts"] for r in hybrid_runs) / max(len(hybrid_runs), 1)
    n_seeds = len({r["seed"] for r in results})
    scenario_runs = runs_per_seed * n_seeds
    llm_cost = scenario_runs * windows_per_run * (480 * p_in + 60 * p_out)
    triage_cost = alerts_per_run * PER_ALERT_PROMPT_TOKENS * p_in  # sampled once
    total = (llm_cost + triage_cost) * 2.0
    return dict(scenario_runs=scenario_runs,
                windows_per_run=round(windows_per_run, 1),
                prompt_tokens_per_call=480,
                completion_tokens_per_call=60,
                mean_alerts_per_run=round(alerts_per_run),
                hybrid_llm_usd=round(llm_cost, 2),
                triage_reference_usd=round(triage_cost, 2),
                safety_factor=2.0,
                upper_bound_usd=round(total, 2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    run_all(args.data, args.out)
