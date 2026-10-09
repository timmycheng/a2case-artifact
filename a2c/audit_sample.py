"""Session-to-incident mapping audit: stratified sample preparation
(stage-2 prep; advisor next_step.md, addition 2).

Every ARI/purity number in the paper is measured against the
session-derived reference partition: an attack alert belongs to the
incident whose documented window and topology it was drawn from (or, for
real EVE input, falls into). This script prepares the human audit of that
mapping. It draws a stratified random sample of ~50 *mapped attack
alerts* (the unit of the session-to-incident mapping: one alert = one
mapping assignment), stratified over the dataset's attack signature
families (15: the 14 attack labels, with Infiltration spanning the TROJAN
and POLICY-EXFIL families) plus the disclosed-synthetic D2 twin stratum,
and emits an audit worksheet for manual verdicts together with a sampling
manifest.

For every sampled alert the worksheet records: the alert as an analyst
would see it, the reference mapping (assigned incident), the documented
session facts it should match (window, attacker IPs, victim, port), and
programmatic consistency checks. The auditor records a verdict per row;
mapping precision = correct verdicts / sampled rows. A companion
sensitivity sweep (audit_sensitivity.py, stage 2) then perturbs the
reference partition at the observed error rate and re-scores the main
metrics.

Sampling design (fixed before running): seed 20260921; minimum 3 alerts
per stratum (15 strata), the remaining 5 allocated to the 4 largest
strata by population (deterministic tie-break by name); uniform sampling
within strata from the seed-17 scenario builds the paper evaluates
(ratio-1.0 streams for Tue/Wed/Thu/Fri plus D2 twins). Benign
false-positive alerts carry no mapping claim and are out of scope.

Entry point:
    python -m a2c.audit_sample --data <csv-dir> --out results/audit
"""

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone

from . import schedule, scenarios
from .arms import _windowize  # noqa: F401  (import parity with runners)
from .ingest import build_stream
from .scenarios import DAY_KEYS, FRIDAY_MEMBERS, THURSDAY_MEMBERS

SEED = 20260921
N_TARGET = 50
MIN_PER_STRATUM = 3

EPOCH = datetime(1970, 1, 1)


def _iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S")


def build_day_stream(zf, day_key, member):
    """The seed-17 ratio-1.0 stream for one scenario day (the attack-alert
    population is identical across the A/B noise arms of the same day)."""
    return build_stream(zf, member, day_key, 1.0, seed=17)


def population(zf):
    """Stratum populations: (family stratum -> attack alerts) across the
    evaluated days, plus the synthetic D2 twin stratum."""
    strata = {}
    provenance = {}   # alert_id -> (scenario_day, stream)
    day_members = [
        ("tuesday", "Tuesday-WorkingHours.pcap_ISCX.csv"),
        ("wednesday", "Wednesday-workingHours.pcap_ISCX.csv"),
        ("thursday_am", THURSDAY_MEMBERS[0]),
        ("thursday_pm", THURSDAY_MEMBERS[1]),
        ("friday_am", FRIDAY_MEMBERS[0]),
        ("friday_ps", FRIDAY_MEMBERS[1]),
        ("friday_ddos", FRIDAY_MEMBERS[2]),
    ]
    for day_key, member in day_members:
        alerts, _inc, _st = build_day_stream(zf, day_key, member)
        for a in alerts:
            if a.is_attack:
                strata.setdefault(a.family, []).append(a)
                provenance[a.alert_id] = day_key
    # D2 twins: synthetic same-family concurrent campaign (disclosed)
    d2 = scenarios.build_twins(zf, seed=17)
    twins = [a for a in d2.alerts if a.gt_incident == "dos-hulk-twin"]
    strata["DOS-HULK-twin (synthetic D2)"] = twins
    for a in twins:
        provenance[a.alert_id] = "wednesday (D2 twin)"
    return strata, provenance


def allocate(strata, rng):
    """Min-3 per stratum, remainder to the largest strata (deterministic)."""
    names = sorted(strata)
    alloc = {n: MIN_PER_STRATUM for n in names}
    rest = N_TARGET - MIN_PER_STRATUM * len(names)
    if rest > 0:
        biggest = sorted(names, key=lambda n: (-len(strata[n]), n))
        for i in range(rest):
            alloc[biggest[i % len(biggest)]] += 1
    return alloc


def doc_facts(family):
    """Documented session facts for a family's signature family -> the
    dataset attack label(s) it emulates (signatures.SIGNATURES keys)."""
    from .signatures import SIGNATURES
    labels = [lab for lab, variants in SIGNATURES.items()
              if any(v[1] == family for v in variants)]
    return labels


def worksheet_rows(sampled, provenance, incident_context):
    from .signatures import SIGNATURES
    fam_to_label = {}
    for lab, variants in SIGNATURES.items():
        for tmpl, fam, _sev in variants:
            fam_to_label[fam] = lab
    rows = []
    for stratum, a in sampled:
        if "twin" in stratum:
            meta = schedule.ATTACKS["DoS Hulk"]
            doc = dict(window=meta["window"], srcs=["205.174.165.70"],
                       dst=meta["dst"], port=meta["port"])
            label = "DoS Hulk twin (synthetic)"
            assigned = a.gt_incident
        else:
            label = fam_to_label.get(a.family, "?")
            meta = schedule.ATTACKS[label]
            doc = dict(window=meta["window"], srcs=meta["src"],
                       dst=meta["dst"], port=meta["port"])
            assigned = a.gt_incident
        s, e = doc["window"]
        # schedule windows are naive datetimes the harness treats as UTC
        # (ingest._ts does naive arithmetic); .timestamp() on a naive
        # datetime would assume local time and shift everything by the
        # machine's UTC offset.
        s_ts = s.replace(tzinfo=timezone.utc).timestamp()
        e_ts = e.replace(tzinfo=timezone.utc).timestamp()
        checks = dict(
            ts_in_window=bool(s_ts <= a.ts <= e_ts),
            src_documented=a.src in doc["srcs"],
            dst_documented=a.dst == doc["dst"],
            port_documented=(None if doc["port"] == 0
                             else a.dst_port == doc["port"]),
        )
        phase = ""
        if label == "Infiltration":
            for pname, (ps, pe) in schedule.INFILTRATION_PHASES:
                ps_ts = ps.replace(tzinfo=timezone.utc).timestamp()
                pe_ts = pe.replace(tzinfo=timezone.utc).timestamp()
                if ps_ts <= a.ts <= pe_ts:
                    phase = pname
                    break
        # what the human must judge on this row, in plain words
        if not all([checks["ts_in_window"], checks["src_documented"],
                    checks["dst_documented"]]):
            hint = ("HARD FAIL: a mechanical check failed -- this would be "
                    "a mapping-rule bug; double-check the fields")
        elif label == "Infiltration":
            hint = (f"phase={phase or 'OUTSIDE-KNOWN-PHASES'}: does the "
                    "alert fit its infiltration phase?")
        elif "twin" in stratum:
            hint = ("synthetic twin: attacker must be .70 (not .73), "
                    "same window/victim as DoS Hulk")
        elif checks["port_documented"] is False:
            hint = (f"port {a.dst_port} != documented {doc['port']}: same "
                    "session despite the service-port difference?")
        elif checks["port_documented"] is None:
            hint = ("no single documented port: judge by attacker, victim, "
                    "window, and signature semantics")
        else:
            hint = ("all checks pass: confirm the signature text reads "
                    "like this incident's activity")
        # mechanical pre-verdict: only all-clear rows are auto-filled;
        # anything flagged or semantic stays with the human
        suggested = ("correct" if all([checks["ts_in_window"],
                                       checks["src_documented"],
                                       checks["dst_documented"],
                                       checks["port_documented"] is not False])
                     else "")
        # what the assignment groups this alert with: up to 5 sibling
        # alerts of the same incident (time span + endpoint variety), so
        # the auditor can judge whether the grouping is coherent
        sibs = incident_context.get(assigned, [])
        context = " | ".join(
            f"{_iso(b.ts)[11:19]} {b.src}->{b.dst}:{b.dst_port} {b.family}"
            for b in sibs[:5]) or "(sole alert of this incident)"
        rows.append(dict(
            stratum=stratum, scenario_day=provenance[a.alert_id],
            alert_id=a.alert_id, family=a.family, sid=a.sid,
            ts_utc=_iso(a.ts), src=a.src, dst=a.dst, dst_port=a.dst_port,
            proto=a.proto, text=a.text, assigned_incident=assigned,
            doc_label=label, doc_window=f"{_iso(s_ts)}..{_iso(e_ts)}",
            doc_attackers=";".join(doc["srcs"]), doc_victim=doc["dst"],
            doc_port=doc["port"], phase=phase,
            check_ts_in_window=checks["ts_in_window"],
            check_src_documented=checks["src_documented"],
            check_dst_documented=checks["dst_documented"],
            check_port_documented=checks["port_documented"],
            judgment_hint=hint, suggested_verdict=suggested,
            incident_context=context,
            verdict="", auditor_notes=""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/audit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    zf = args.data

    strata, provenance = population(zf)
    rng = random.Random(SEED)
    alloc = allocate(strata, rng)
    sampled = []
    for name in sorted(strata):
        pool = strata[name]
        k = min(alloc[name], len(pool))
        sampled += [(name, a) for a in rng.sample(pool, k)]
    # sibling-alert context per incident: what the assignment groups
    # together, drawn from the same builds (up to 8 per incident, spread)
    incident_context = {}
    by_inc = {}
    for name in sorted(strata):
        for a in strata[name]:
            if a.gt_incident:
                by_inc.setdefault(a.gt_incident, []).append(a)
    for inc, alerts in by_inc.items():
        alerts.sort(key=lambda x: x.ts)
        step = max(1, len(alerts) // 8)
        incident_context[inc] = alerts[::step]
    rows = worksheet_rows(sampled, provenance, incident_context)

    csv_path = os.path.join(args.out, "audit_worksheet.csv")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    except PermissionError:
        # the old sheet is open in Excel or held by the sync client --
        # never clobber it; write a versioned filename instead
        csv_path = os.path.join(args.out, "audit_worksheet_v2.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    manifest = dict(
        date="2026-09-21", seed=SEED, target_n=N_TARGET,
        unit="one mapped attack alert (one session-to-incident assignment)",
        precision_definition=("fraction of sampled alerts whose assigned "
                              "incident the auditor judges correct"),
        verdicts=["correct", "wrong-incident", "ambiguous"],
        build=dict(scenario_seed=17, noise_ratio=1.0,
                   twins="scenarios.build_twins seed 17 (synthetic D2)"),
        strata={n: dict(population=len(strata[n]), sampled=alloc[n])
                for n in sorted(strata)},
        out_of_scope="benign false-positive alerts (no mapping claim)",
    )
    with open(os.path.join(args.out, "audit_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    n_auto = sum(1 for r in rows if r["suggested_verdict"])
    readme = f"""# 映射审计工作表（生成于 {manifest['date']}）

## 你在审什么（一句话版）

不是审"这条告警是不是攻击"——攻击与否由数据集标签定义，那是基准的定义。
你审的是**映射**：这条告警被分到 `assigned_incident` 这个事件，依据是否
成立。判错的映射会让论文所有 ARI 指标失真，这个审计把"映射是假设"变成
"有界近似"。

## 怎么审（预期 ~30-45 分钟）

1. 打开 `audit_worksheet.csv`，先按 `suggested_verdict` 排序。
2. **预填了 "correct" 的行（{n_auto} 行）**：四项机械检查全过、端口也匹配。
   你只需抽查三分之一左右，扫一眼 `text` 签名语义和 `incident_context`
   （同事件同伴告警）是否一致，没有异议就保持 "correct"。
3. **`suggested_verdict` 为空的行**：这是你的主要工作。每行都有
   `judgment_hint` 告诉你判断点：
   - 端口不匹配行：同一攻击者、同一受害机、同一窗口内的告警，服务端口
     不同（如 Bot 的 C2 端口 443 vs 实际流端口 52726）算不算同一事件？
     你的判断定义了 precision 的容差。
   - Infiltration 行：看 `phase` 列，告警语义是否符合该阶段（meta/COOL
     磁盘/Dropbox）。
   - 双胞胎行（DOS-HULK-twin）：攻击者必须是 .70 而非 .73；窗口/受害机
     与 DoS Hulk 相同。
   - "HARD FAIL" 行：机械检查失败=映射规则 bug，需要复核字段。
4. 把你的最终判定填进 `verdict`：correct / wrong-incident / ambiguous，
   有话想说填 `auditor_notes`。

判完把结果发回来：precision = #correct / #{len(rows)}，加上你对
`ambiguous` 行的处置意见。误差率 (1-precision) 喂给敏感性扫描
（`audit_sensitivity.py`，已按 e=0–15% 预跑完毕）。

## 抽样与构建（可复核性）

- {len(rows)} 条映射告警，{len(strata)} 个分层（15 个攻击签名家族 +
  合成 D2 双胞胎层；Infiltration 横跨 TROJAN/POLICY-EXFIL 两族），每层
  至少 {MIN_PER_STRATUM} 条，余量给最大层，seed {SEED}，可复现。
- 构建：seed-17 ratio-1.0 流 + D2 双胞胎——与论文评测用同一批构建，
  因此审计覆盖了所有指标所用的参考划分。
- 良性误报告警不审计（它们没有映射主张）。PortScan/DDoS 行 doc_port=0
  表示数据集未记载单一服务端口，端口检查仅供参考。

## 局限（如实声明）

仿真路径上，时间戳/端点是按文档规则构造的，所以机械检查大部分会通过——
这个审计度量的正是"构造规则 + 文档窗口"这套映射的例外与边界情况
（泄漏、双胞胎分离、多阶段归属、端口容差），以及在真实 EVE 重挂路径上
同一套规则将要面对的判断类型。
"""
    with open(os.path.join(args.out, "README.md"), "w",
              encoding="utf-8") as f:
        f.write(readme)

    print(f"sampled {len(rows)} alerts over {len(strata)} strata -> {csv_path}")
    for n in sorted(strata):
        print(f"  {n:34s} pop={len(strata[n]):5d} sampled={alloc[n]}")


if __name__ == "__main__":
    main()
