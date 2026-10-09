import json

# tab:llm: Qwen block (ARI, Valid) + GLM block (ARI, Under, Valid,
# norm USD/10k, actual USD/10k), seed 17, per scenario
TEX = {
    # scen: (q_ari, q_valid, g_ari, g_under, g_valid, norm_usd, act_usd)
    "A1": (1.000, "105/105", 0.623, 1.00, "92/105", 0.130, 0.975),
    "A2": (1.000, "107/107", 0.547, 1.00, "88/107", 0.068, 0.512),
    "A3": (0.998, "102/103", 0.778, 0.67, "65/103", 0.113, 0.845),
    "B2": (1.000, "108/108", 0.484, 1.00, "86/108", 0.017, 0.130),
    "B3": (0.996, "107/108", 0.961, 0.33, "74/108", 0.025, 0.186),
    "E":  (1.000, "101/101", 0.414, 0.50, "67/101", 0.307, 2.291),
    "D2": (0.886, "107/107", 0.600, 0.67, "83/107", 0.070, 0.521),
}

local = {}
for r in json.load(open("results/round3_local.json")):
    if r.get("part") == "original_local" and r["arm"] == "hybrid_llm_0.5B":
        local[r["scenario"]] = r
glm = {}
for r in json.load(open("results/round3_glm.json")):
    if r.get("part") == "original_glm" and r["arm"] == "hybrid_llm_glm5.3":
        glm[r["scenario"]] = r

MINI_IN, MINI_OUT = 0.15e-6, 0.60e-6
GLM_IN, GLM_OUT = 1.40e-6, 4.40e-6

problems = []
for scen, (q_ari, q_val, g_ari, g_und, g_val, norm, act) in TEX.items():
    q = local[scen]
    checks = [
        ("q_ari", q["ari"], q_ari, 0.0005),
        ("q_valid", f"{q['llm_valid']}/{q['llm_calls']}", q_val, None),
    ]
    g = glm.get(scen)
    if g:
        checks += [
            ("g_ari", g["ari"], g_ari, 0.0005),
            ("g_under", g["under"], g_und, 0.005),
            ("g_valid", f"{g['llm_valid']}/{g['llm_calls']}", g_val, None),
            ("norm_usd", (g["prompt_tokens"] * MINI_IN
                          + g["completion_tokens"] * MINI_OUT)
             / g["n_alerts"] * 10000, norm, 0.0005),
            ("act_usd", (g["prompt_tokens"] * GLM_IN
                         + g["completion_tokens"] * GLM_OUT)
             / g["n_alerts"] * 10000, act, 0.0005),
        ]
    for name, actual, shown, tol in checks:
        if tol is None:
            if str(actual) != str(shown):
                problems.append(f"{scen}/{name}: tex={shown} json={actual}")
        elif abs(actual - shown) > tol:
            problems.append(f"{scen}/{name}: tex={shown} json={actual:.4f}")

print("PROBLEMS:" if problems else "tab:llm: ALL CELLS MATCH (incl cost arith)")
for p in problems:
    print(" ", p)
