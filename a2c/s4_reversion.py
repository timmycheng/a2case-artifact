import os
"""S4 reversion mechanism experiments (stage 3 of the journal plan).

The Table-6 anomaly under study: on the twin-injection stress test D2 the
0.5B instruction model reaches ARI 1.000 at summary level S3 (per-family
source breakdown) but reverts to 0.886 at S4 (the full (family, dst, src)
cross-tabulation). Two separately falsifiable hypotheses (advisor
next_step.md, addition 3):

  A -- token-length dilution. The S4 summary block is much longer than S3;
       the discriminative signal drowns in serialized context. Test
       condition S4M: the *same* S4 summary (identical fields, values,
       ordering and truncations) re-serialized as a compact matrix of
       plain-text lines instead of JSON. If S4M recovers the S3 result,
       the reversion is a length/attention artifact, not a semantic one.

  B -- cross-tab pseudo-correlation. What breaks the model is the
       cross-tabulation content itself (redundant/confounding
       co-occurrences), not its length. Test condition S3P: S3 plus exactly
       one field, per-family distinct-source counts (the attacker-count
       signal in its purest form), *without* the cross-tab, serialized in
       the same JSON style as S3/S4. If S3P holds the S3 result while S4
       reverts, the reversion is content-driven (signal dilution by
       confounded fields), not length-driven.

Design invariants (fixed before running):
  * the prompt template, decoding, JSON validation and fallback semantics
    are identical across conditions -- only the summary block changes;
  * four conditions: S3 and S4 baselines re-run in-run (reproducibility
    anchor against the round3 rows), S3P (Hypothesis B), S4M (A);
  * summary block token counts are measured with the deployed model's
    tokenizer and reported per condition -- the Hypothesis-A mechanism
    statistic (same information, fewer tokens);
  * decoding is deterministic (local greedy; GLM temperature 0).

Entry point:
    python -m a2c.s4_reversion --data <csv-dir> --out results/s4_reversion.json \
        --gen local|glm|both
"""

import argparse
import json
import random
from collections import defaultdict

from . import metrics, scenarios
from .arms import _windowize
from .classics import GLMPolicyGen, run_hybrid_precomputed, stream_summary


GLM_KEY = os.environ.get("GLM_API_KEY", "")  # set GLM_API_KEY to run the GLM conditions
GLM_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"
QWEN = os.environ.get("A2C_QWEN_DIR", "../data/models/Qwen2.5-0.5B-Instruct")


# ---------------------------------------------------------------------------
# Summary-block builders. Each returns the exact text placed in the prompt's
# "Window summary:" slot: dict -> generator-standard JSON, str -> verbatim.
# ---------------------------------------------------------------------------

def s3plus_summary(window_alerts) -> dict:
    """S3 plus one field: family_attacker_counts (distinct src IPs per
    family), for exactly the families family_sources already reports. The
    full distinct count is deliberate: family_sources truncates to the 4
    most common sources, so the cardinality signal itself is new
    information, and it is the discriminating feature for concurrent
    same-family campaigns (two attackers on one victim)."""
    s = stream_summary(window_alerts, level=3)
    per_fam = defaultdict(set)
    for a in window_alerts:
        per_fam[a.family].add(a.src)
    s["family_attacker_counts"] = {f: len(per_fam[f])
                                   for f in s.get("family_sources", {})}
    return s


def s4_matrix_text(window_alerts) -> str:
    """The S4 summary re-serialized as compact text. Consumes the standard
    level-4 dict; every field, value, ordering and truncation is the same,
    only the serializer differs (no JSON quoting/bracket overhead)."""
    s = stream_summary(window_alerts, level=4)
    lines = [f"n={s['n']}"]
    lines.append("families: " + (", ".join(f"{f}={n}"
                                           for f, n in s["families"]) or "-"))
    lines.append("top_dsts: " + (", ".join(f"{d}={n}"
                                           for d, n in s["top_dsts"]) or "-"))
    lines.append("top_pairs: " + (", ".join(f"{p}={n}"
                                            for p, n in s["top_pairs"]) or "-"))
    if "top_srcs" in s:
        lines.append("top_srcs: " + (", ".join(f"{p}={n}"
                                               for p, n in s["top_srcs"])
                                     or "-"))
    if "family_sources" in s:
        parts = []
        for f, srcs in s["family_sources"].items():
            inner = ", ".join(f"{p}={n}" for p, n in srcs.items())
            parts.append(f"{f} <- {inner}" if inner else f)
        lines.append("family_sources: " + ("; ".join(parts) or "-"))
    if "family_dst_src" in s:
        lines.append("cross-tab family|dst|src=n (top):")
        for fam, dst, src, n in s["family_dst_src"]:
            lines.append(f"  {fam}|{dst}|{src}={n}")
    return "\n".join(lines)


CONDITIONS = [
    # (condition id, builder) -- builder(window) -> dict | str
    ("S3", lambda w: stream_summary(w, level=3)),
    ("S4", lambda w: stream_summary(w, level=4)),
    ("S3P", s3plus_summary),
    ("S4M", s4_matrix_text),
]

HYPOTHESIS = {"S3": "baseline (paper: 0.5B ARI 1.000)",
              "S4": "baseline (paper: 0.5B ARI 0.886, the reversion)",
              "S3P": "Hypothesis B (content, not length)",
              "S4M": "Hypothesis A (length, not content)"}


def _block_for_gen(builder, window, gen: str):
    """Materialize the summary block in the generator's deployed style:
    the local model saw indent=1 JSON, the GLM API path compact JSON."""
    s = builder(window)
    if isinstance(s, str):
        return s
    return json.dumps(s, indent=1) if gen == "local" else json.dumps(s)


def measure_summary_tokens(windows, tok, gen: str) -> dict:
    """Tokenizer-measured size of each condition's summary block across
    all D2 windows (the Hypothesis-A mechanism statistic)."""
    out = {}
    for name, builder in CONDITIONS:
        counts = []
        for w in windows:
            block = _block_for_gen(builder, w, gen)
            counts.append(len(tok.encode(block)))
        counts.sort()
        q = lambda p: counts[min(len(counts) - 1, int(p * len(counts)))]
        out[name] = dict(mean=sum(counts) / len(counts),
                         p50=q(0.5), p90=q(0.9), max=counts[-1],
                         n_windows=len(counts))
    return out


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_local(zf, out_rows, tok_path=QWEN):
    from .locals import LocalLLMPolicyGen
    sc = scenarios.build_twins(zf, seed=17)
    windows = _windowize(sc.alerts, 300)
    n = len(sc.alerts)

    tok_stats = None
    g = LocalLLMPolicyGen(tok_path)
    for name, builder in CONDITIONS:
        # per-condition token accounting: the generator's counters are
        # cumulative, so snapshot before each condition and report deltas
        pre_ptok, pre_ctok, pre_calls, pre_fb = (
            g.prompt_tokens, g.completion_tokens, g.calls, g.fallbacks)
        pols = []
        for w in windows:
            user = _local_user(_block_for_gen(builder, w, "local"))
            pols.append(g.generate_policy_text(user, n_window=len(w)))
        cond = dict(prompt_tokens=g.prompt_tokens - pre_ptok,
                    completion_tokens=g.completion_tokens - pre_ctok,
                    calls=g.calls - pre_calls,
                    fallbacks=g.fallbacks - pre_fb)
        _append_row(out_rows, sc, f"qwen0.5B_{name}", name, g, pols, n,
                    prompt_tokens=cond["prompt_tokens"],
                    completion_tokens=cond["completion_tokens"],
                    calls=cond["calls"], fallbacks=cond["fallbacks"])
        print(f"  [local {name}] ari="
              f"{out_rows[-1]['ari']:.3f} "
              f"ptok/call={cond['prompt_tokens'] / max(1, cond['calls']):.0f}",
              flush=True)
        if tok_stats is None:  # tokenizer already loaded on the generator
            tok_stats = measure_summary_tokens(windows, g.tok, "local")
    return tok_stats


def _local_user(block: str) -> str:
    from .locals import _LLM_USER
    return _LLM_USER.format(summary=block)


def glm_api_ok(key: str, timeout: int = 30) -> bool:
    """One-token probe so a quota-blocked key fails fast instead of
    grinding the per-call retry ladder for hours (429 observed
    2026-09-21/22 on this key)."""
    import urllib.error
    import urllib.request
    body = dict(model="glm-5.3", messages=[{"role": "user",
                                            "content": "ok"}],
                max_tokens=1, temperature=0)
    req = urllib.request.Request(
        f"{GLM_BASE}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
        return True
    except urllib.error.HTTPError as e:
        print(f"glm probe: HTTP {e.code} -- run blocked, not starting",
              flush=True)
        return False
    except Exception as exc:  # noqa: BLE001 - network: let caller retry
        print(f"glm probe: {exc!r} -- proceeding", flush=True)
        return True


def run_glm(zf, out_rows, model="glm-5.3"):
    sc = scenarios.build_twins(zf, seed=17)
    windows = _windowize(sc.alerts, 300)
    n = len(sc.alerts)
    for name, builder in CONDITIONS:
        g = GLMPolicyGen(GLM_KEY, model=model, base_url=GLM_BASE)
        summaries = [_block_for_gen(builder, w, "glm") for w in windows]
        pols = g.policies_parallel(windows, workers=2, summaries=summaries)
        # honest-abort guard: if every call failed at the API layer, the
        # condition's ARI would be a fallback artifact, not model behavior
        if g.calls == 0 or g.fallbacks == g.calls:
            raise SystemExit(
                f"glm condition {name}: all {g.calls or '0'} calls failed "
                "(rate limit/auth?) -- aborting without recording results")
        _append_row(out_rows, sc, f"{model}_{name}", name, g, pols, n,
                    prompt_tokens=g.prompt_tokens,
                    completion_tokens=g.completion_tokens,
                    calls=g.calls)
        print(f"  [glm {name}] ari={out_rows[-1]['ari']:.3f} "
              f"valid={g.calls - g.fallbacks}/{g.calls}", flush=True)
        json.dump(out_rows, open(_ROWS_PATH, "w"), indent=1)


_ROWS_PATH = None


def _append_row(out_rows, sc, arm, cond, gen, pols, n, prompt_tokens=None,
                completion_tokens=None, calls=None, fallbacks=None):
    """One result row. The token/call/fallback values default to the
    generator's cumulative counters; per-condition runners pass the
    condition-scoped values."""
    res = run_hybrid_precomputed(sc.alerts, pols)
    ptok = gen.prompt_tokens if prompt_tokens is None else prompt_tokens
    ctok = (gen.completion_tokens if completion_tokens is None
            else completion_tokens)
    ncalls = gen.calls if calls is None else calls
    nfb = gen.fallbacks if fallbacks is None else fallbacks
    res["prompt_tokens"] = ptok
    res["completion_tokens"] = ctok
    ev = metrics.evaluate(res["cases"], n, len(sc.incidents))
    out_rows.append(dict(
        part="s4_reversion", scenario="D2", arm=arm, condition=cond,
        hypothesis=HYPOTHESIS[cond], seed=17,
        n_alerts=n, n_incidents=len(sc.incidents), **ev,
        llm_valid=ncalls - nfb, llm_calls=ncalls,
        prompt_tokens=ptok, completion_tokens=ctok,
        prompt_tokens_per_call=ptok / max(1, ncalls),
        completion_tokens_per_call=ctok / max(1, ncalls),
        fallbacks=nfb,
        dims_hist=json.dumps(sorted({tuple(t["dims"])
                                     for t in res["policy_trace"]}))))
    if _ROWS_PATH:
        json.dump(out_rows, open(_ROWS_PATH, "w"), indent=1)


def main():
    global _ROWS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/s4_reversion.json")
    ap.add_argument("--gen", choices=["local", "glm", "both"], default="both")
    args = ap.parse_args()
    _ROWS_PATH = args.out
    zf = args.data
    out_rows = []
    json.dump(out_rows, open(args.out, "w"), indent=1)

    if args.gen in ("local", "both"):
        tok_stats = run_local(zf, out_rows)
        out_rows.append(dict(part="s4_reversion_summary_tokens", gen="local",
                             tokenizer="Qwen2.5-0.5B-Instruct", **tok_stats))
        json.dump(out_rows, open(args.out, "w"), indent=1)
    if args.gen in ("glm", "both"):
        if glm_api_ok(GLM_KEY):
            run_glm(zf, out_rows)
        else:
            print("skipping glm conditions (probe failed)", flush=True)
    print("s4_reversion complete:", len(out_rows), "rows", flush=True)


if __name__ == "__main__":
    main()
