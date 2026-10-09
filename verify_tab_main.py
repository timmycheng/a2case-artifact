import json
import re

tex = open("tables/table_main_wrap.tex", encoding="utf-8").read()
rows = json.load(open("results/results.json"))["results"]
grid = {}
for r in rows:
    grid[(r["scenario"], r["arm"], r.get("seed", 17))] = r

ARM_MAP = {"Static rules": "static", "Embedding": "embedding",
           "Generated (PSH)": "hybrid_psh"}
NUM = r"(?:\\textbf\{)?([\d.]+)\}?(?:\$\^{\w+}\$)?"
body = re.findall(
    r"^([AEB]\d?)(?:\$\^{\w+}\$)?\s+&\s+(.+?)\s+&\s+" + NUM +
    r"\s+&\s+" + NUM + r"\s+&\s+" + NUM + r"\s+&\s+" + NUM +
    r"\s+&\s+(\d+)\s+&\s+([\d.]+)",
    tex, re.M)

D2_ARM_MAP = {
    "Static rules (\\texttt{src,dst})": ("static", ("src", "dst")),
    "Static rules (\\texttt{family})": ("static", ("family",)),
    "Static rules (\\texttt{src,family})": ("static", ("src", "family")),
    "Embedding": ("embedding", None),
    "Generated (PSH)": ("hybrid_psh", None),
    "Generated (LLM 0.5B)": ("llm", None),
}

problems2 = []
d2_grid = {}
for r in rows:
    if r["scenario"] == "D2":
        d2_grid[r["arm"]] = r
llm_d2 = None
for src in ("results/round2.json", "results/results.json"):
    try:
        for r in json.load(open(src)).get("results", []):
            if r.get("scenario") == "D2" and "llm" in r.get("arm", ""):
                llm_d2 = r
    except Exception:
        pass
try:
    for r in json.load(open("results/round3_local.json")):
        if r.get("scenario") == "D2" and r.get("part") == "original_local":
            llm_d2 = r if r["arm"] == "hybrid_llm_0.5B" else llm_d2
except Exception:
    pass

for scen, arm, ari, pur, under, over, compr, usd in body:
    arm = arm.strip()
    if arm in ARM_MAP:
        r = grid.get((scen, ARM_MAP[arm], 17))
        src_note = "results.json"
    elif arm in D2_ARM_MAP and scen == "D2":
        key = D2_ARM_MAP[arm][0]
        r = d2_grid.get(key) if key != "llm" else llm_d2
        src_note = "D2 source"
        if key == "static" and D2_ARM_MAP[arm][1]:
            # tab:main's D2 static rows are specific dimension sets from
            # the no-fixed-winner sweep; verify against static_sweep
            continue
    else:
        continue
    if r is None:
        problems2.append(f"{scen}/{arm}: no source row")
        continue
    for key, shown in [("ari", float(ari)), ("purity", float(pur)),
                       ("under", float(under)), ("over", float(over)),
                       ("compression", float(compr))]:
        actual = r[key]
        tol = 0.005001 if key != "compression" else 0.5
        if abs(actual - shown) > tol:
            problems2.append(
                f"{scen}/{arm}/{key} [{src_note}]: tex={shown} "
                f"json={actual:.4f}")

print(f"all rows: {len(body)} parsed")
print("PROBLEMS:" if problems2 else "ALL VERIFIABLE CELLS MATCH")
for p in problems2:
    print(" ", p)
