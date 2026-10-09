"""Partition-faithfulness metrics and cost accounting (L1 level)."""

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def _majority(items):
    counts = {}
    for it in items:
        counts[it] = counts.get(it, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1]) if counts else ("", 0)


def evaluate(partition, n_total_alerts, n_incidents):
    """Evaluate a case partition against case-level ground truth.

    Faithfulness metrics are computed on the attack-alert restriction (the
    proposal's L1 metrics); compression and case counts use the full stream.

    - ari / nmi: case-assignment agreement on attack alerts
    - purity: weighted majority share of ground-truth incidents per case
    - under: share of incidents split across >1 case (missed groupings)
    - over: share of attack-bearing cases spanning >1 incident (over-grouping)
    - compression: alerts per case on the full stream
    - benign_contam: false positives absorbed into attack-bearing cases,
      per attack alert
    """
    case_of = {}
    labels_pred, labels_true = [], []
    purity_num = 0
    benign_in_attack_cases = 0
    case_incident_sets = []
    attack_n = 0

    for ci, case in enumerate(partition):
        inc_here = []
        for a in case:
            case_of[a.alert_id] = ci
            if a.is_attack:
                attack_n += 1
                labels_pred.append(ci)
                labels_true.append(a.gt_incident)
                inc_here.append(a.gt_incident)
            else:
                if inc_here:
                    pass
        if inc_here:
            maj, maj_n = _majority(inc_here)
            purity_num += maj_n
            benign_in_attack_cases += sum(1 for a in case if not a.is_attack)
            case_incident_sets.append(set(inc_here))

    total_cases = len(partition)
    fp_n = n_total_alerts - attack_n

    if attack_n == 0:
        return dict(ari=float("nan"), nmi=float("nan"), purity=float("nan"),
                    under=0.0, over=0.0, compression=0.0,
                    cases_per_incident=0.0, benign_contam=0.0,
                    cases_total=total_cases)

    ari = float(adjusted_rand_score(labels_true, labels_pred))
    nmi = float(normalized_mutual_info_score(labels_true, labels_pred))
    purity = purity_num / attack_n

    incident_cases = {}
    for a_case_i, inc in zip(labels_pred, labels_true):
        incident_cases.setdefault(inc, set()).add(a_case_i)
    under_rate = (sum(1 for s in incident_cases.values() if len(s) > 1)
                  / len(incident_cases))
    over_rate = (sum(1 for s in case_incident_sets if len(s) > 1)
                 / len(case_incident_sets)) if case_incident_sets else 0.0

    return dict(ari=ari, nmi=nmi, purity=purity,
                under=under_rate, over=over_rate,
                compression=n_total_alerts / total_cases if total_cases else 0.0,
                cases_total=total_cases,
                cases_per_incident=total_cases / n_incidents,
                benign_contam=benign_in_attack_cases / attack_n)


def cost_row(res, n_alerts, price_in=0.15 / 1e6, price_out=0.60 / 1e6,
             est_in_tok_per_alert=0.0):
    """Cost accounting per 10k alerts (default prices: gpt-4o-mini list).

    For the per-alert triage reference, pass est_in_tok_per_alert (the
    fixed per-alert prompt size) — the analytic cost model of the proposal,
    c_x = K_x u_bar / n + V_x / n, evaluated at K=1, V=0.
    """
    usd = (res["prompt_tokens"] * price_in + res["completion_tokens"] * price_out)
    usd += est_in_tok_per_alert * n_alerts * price_in
    return dict(policy_calls=res["policy_calls"],
                prompt_tokens=res["prompt_tokens"],
                completion_tokens=res["completion_tokens"],
                usd_total=usd,
                usd_per_10k=usd / n_alerts * 10_000 if n_alerts else 0.0,
                wall=res["wall"])
