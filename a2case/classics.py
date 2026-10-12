"""Round-3 machinery: oracle policy selector, Julisch-style classic baseline,
summary variants (information ablation), GLM client with parallel precompute,
parametric scenario generator, and paired statistics."""

import json
import random
import time
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import adjusted_rand_score

from . import schedule
from .policies import Policy, policy_key, VALID_DIMS, DEFAULT_STATIC_POLICY
from .signatures import SIGNATURES, _sid_for, Alert

# ---------------------------------------------------------------------------
# Candidate policy space (shared by oracle and generators)
# ---------------------------------------------------------------------------
DIM_SETS = [
    ("src,dst", ["src", "dst"]),
    ("src,dst,port", ["src", "dst", "dst_port"]),
    ("dst,port", ["dst", "dst_port"]),
    ("family", ["family"]),
    ("family,dst", ["family", "dst"]),
    ("sid", ["sid"]),
    ("src,family", ["src", "family"]),
]
CANDIDATES = [(dims, w) for label, dims in DIM_SETS for w in (5, 10)]


def _local_partition(alerts, dims, window_min):
    """Partition window alerts under a candidate policy (fresh episodes)."""
    open_by_key = {}
    order = []
    w = window_min * 60
    for a in sorted(alerts, key=lambda x: x.ts):
        k = policy_key(a, dims)
        c = open_by_key.get(k)
        if c is not None and a.ts - c[-1].ts <= w:
            c.append(a)
        else:
            c = [a]
            open_by_key[k] = c
            order.append(c)
    return [c for c in order if c]


class OracleGen:
    """Continuity-aware oracle policy selector.

    For each micro-batch window it trials every candidate policy against a
    snapshot of the executor state, scores the *global* ground-truth ARI of
    the stream partition after applying the candidate, keeps the best, and
    restores/commits accordingly. This is the ceiling of per-window policy
    selection from the same candidate space the generators choose from,
    using ground-truth information no real generator has. Greedy (myopic
    beyond the current window) but continuity-aware."""

    def __init__(self, window_min_choices=(5, 10)):
        """window_min_choices: the inactivity timeouts the oracle may pick
        from (default = the paper's candidate space, unchanged). The
        window-sensitivity study (direction C) sweeps this single
        parameter; everything else about the search is identical."""
        self.calls = 0
        self.trace = []
        self.candidates = [(dims, w) for label, dims in DIM_SETS
                           for w in window_min_choices]

    def run(self, alerts, delta_sec=300, seed=17):
        from .policies import DeterministicExecutor
        from .arms import _windowize
        from .metrics import evaluate
        ex = DeterministicExecutor()
        windows = _windowize(alerts, delta_sec)
        total = len(alerts)
        inc = len({a.gt_incident for a in alerts if a.is_attack})
        trace = []
        for wi, w in enumerate(windows):
            best_pol, best_ari, best_part = None, -2.0, None
            for dims, wmin in self.candidates:
                snap = ex.snapshot()
                ex.apply_window(w, Policy(dims=list(dims), window_min=wmin,
                                          source="oracle"))
                part = ex.case_partition()
                ev = evaluate(part, total, inc)
                ari = ev["ari"] if ev["ari"] == ev["ari"] else -1.0
                ex.restore(snap)
                if ari > best_ari + 1e-12:
                    best_pol, best_ari = (dims, wmin), ari
            ex.apply_window(w, Policy(dims=list(best_pol[0]),
                                      window_min=best_pol[1],
                                      source="oracle"))
            self.calls += 1
            trace.append(dict(window=wi, dims=list(best_pol[0]),
                              window_min=best_pol[1], global_ari=best_ari))
        self.trace = trace
        return ex.case_partition()


# ---------------------------------------------------------------------------
# Julisch-style classic baseline (2003), adapted to streaming micro-batches.
# Each case carries a per-dimension generalization LEVEL with a stored value
# (nested hierarchies: exact -> /24 or service -> *). A new alert joins the
# open case requiring the least additional generalization (raise), and the
# joined case's pattern generalizes (monotonically) to cover it.
# ---------------------------------------------------------------------------

def _ip_gen(ip, level):
    if level == 0:
        return ip
    if level == 1:
        return ".".join(ip.split(".")[:3]) + ".0/24"
    return "*"


def _port_gen(port, level):
    if level == 0:
        return str(port)
    if level == 1:
        svc = {80: "web", 443: "web", 8080: "web", 21: "ftp", 22: "ssh",
               445: "smb", 3389: "rdp", 53: "dns"}
        return svc.get(port, "other")
    return "*"


GEN_DIMS = ["src", "dst", "dst_port", "family"]
GEN_LEVELS = {"src": 3, "dst": 3, "dst_port": 3, "family": 2}
MAX_GEN = sum(GEN_LEVELS.values())


def _gen_dim(a, d, level):
    if d == "family":
        return a.family if level == 0 else "*"
    if d == "dst_port":
        return _port_gen(a.dst_port, level)
    return _ip_gen(getattr(a, d), level)


def _gen_val(d, val, level):
    """Generalize a stored (string) value to `level`."""
    if d == "family":
        return val if level == 0 else "*"
    if d == "dst_port":
        try:
            return _port_gen(int(val), level)
        except ValueError:
            return "other" if level == 0 else ("web" if level == 1 and False else "other")                 if level == 0 else "other"
    return _ip_gen(val, level)


def _raise_cost(a, case):
    """(total_raises, per-dim raises) needed for `case` to cover alert `a`:
    minimal r per dimension such that the alert generalized to lv+r equals
    the stored value generalized to lv+r (both sides raise; hierarchies
    nested so the top level always matches). None if impossible."""
    total, raises = 0, {}
    for d in GEN_DIMS:
        lv, val = case["levels"][d], case["values"][d]
        r = None
        for rr in range(0, GEN_LEVELS[d] - lv + 1):
            if _gen_dim(a, d, lv + rr) == _gen_val(d, val, lv + rr):
                r = rr
                break
        if r is None:
            return None
        raises[d] = r
        total += r
    return total, raises


def run_julisch(alerts, delta_sec=300, window_min=10, raise_cap=MAX_GEN):
    """Julisch-style generalization clustering over the stream."""
    t0 = time.perf_counter()
    open_cases = []
    closed = []
    windows, cur, start = [], [], (alerts[0].ts if alerts else 0.0)
    for a in alerts:
        if a.ts - start >= delta_sec:
            windows.append(cur)
            cur, start = [], a.ts
        cur.append(a)
    if cur:
        windows.append(cur)

    for w in windows:
        for a in w:
            best, best_cost, best_raises = None, raise_cap + 1, None
            for c in open_cases:
                if a.ts - c["last_ts"] > window_min * 60:
                    continue
                got = _raise_cost(a, c)
                if got is not None and got[0] < best_cost:
                    best, best_cost, best_raises = c, got[0], got[1]
            if best is None or best_cost > raise_cap:
                c = dict(levels={d: 0 for d in GEN_DIMS},
                         values={d: _gen_dim(a, d, 0) for d in GEN_DIMS},
                         alerts=[a], last_ts=a.ts)
                open_cases.append(c)
            else:
                c = best
                for d, r in best_raises.items():
                    if r:
                        c["levels"][d] += r
                        c["values"][d] = _gen_dim(a, d, c["levels"][d])
                c["alerts"].append(a)
                c["last_ts"] = max(c["last_ts"], a.ts)
        # close expired cases (memory hygiene; inactivity semantics)
        keep = []
        for c in open_cases:
            (keep if w and w[-1].ts - c["last_ts"] <= window_min * 60
             else closed).append(c)
        open_cases = keep
    open_cases.extend(closed)
    partition = [c["alerts"] for c in open_cases if c["alerts"]]
    return dict(cases=partition, wall=time.perf_counter() - t0,
                policy_calls=0, prompt_tokens=0, completion_tokens=0,
                fallbacks=0, policy_trace=[])


def _pattern_generic(a, d, lv):
    if d == "family":
        return a.family if lv == 0 else "*"
    if d == "dst_port":
        return _port_gen(a.dst_port, lv)
    return _ip_gen(getattr(a, d), lv)


# ---------------------------------------------------------------------------
# Summary variants for the information-sufficiency ablation
# ---------------------------------------------------------------------------

def stream_summary(window_alerts, level=1):
    """Level 1: family counts + top dsts + top pairs (the original).
    Level 2: + top source counts.  Level 3: + per-family source breakdown.
    Level 4: + explicit (family, dst, src) cross-tab."""
    fam = Counter(a.family for a in window_alerts)
    dst = Counter(a.dst for a in window_alerts)
    pair = Counter(f"{a.src}>{a.dst}" for a in window_alerts)
    s = dict(n=len(window_alerts),
             families=sorted(fam.items(), key=lambda kv: -kv[1])[:8],
             top_dsts=sorted(dst.items(), key=lambda kv: -kv[1])[:6],
             top_pairs=sorted(pair.items(), key=lambda kv: -kv[1])[:6])
    if level >= 2:
        src = Counter(a.src for a in window_alerts)
        s["top_srcs"] = sorted(src.items(), key=lambda kv: -kv[1])[:6]
    if level >= 3:
        fam_src = defaultdict(Counter)
        for a in window_alerts:
            fam_src[a.family][a.src] += 1
        s["family_sources"] = {f: dict(c.most_common(4))
                               for f, c in sorted(fam_src.items(),
                                                  key=lambda kv: -sum(kv[1].values()))[:6]}
    if level >= 4:
        cross = Counter((a.family, a.dst, a.src) for a in window_alerts)
        s["family_dst_src"] = [list(k) + [v] for k, v in
                               sorted(cross.items(), key=lambda kv: -kv[1])[:10]]
    return s


# ---------------------------------------------------------------------------
# GLM client (Zhipu open.bigmodel.cn), thinking effort=low, parallel-capable
# ---------------------------------------------------------------------------

class GLMPolicyGen:
    """Frontier-condition policy generator via the GLM API."""

    def __init__(self, key, model="glm-5.3", base_url="https://open.bigmodel.cn/api/paas/v4",
                 effort="low", max_tokens=2048, summary_level=1, seed=17):
        self.key = key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.effort = effort
        self.max_tokens = max_tokens
        self.summary_level = summary_level
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.fallbacks = 0
        self.invalid = 0
        self.wall = 0.0
        self.trace = []

    def _raw_call(self, summary):
        # summary: a dict (compact JSON-serialized) or a preformatted
        # string block; mechanism experiments pass custom string blocks
        # so the prompt template and parsing stay identical.
        block = summary if isinstance(summary, str) else json.dumps(summary)
        body = dict(model=self.model,
                    messages=[
                        {"role": "system",
                         "content": "You respond with ONLY a single JSON object and nothing else."},
                        {"role": "user",
                         "content": _LLM_USER.format(summary=block)}],
                    max_tokens=self.max_tokens, temperature=0,
                    thinking={"type": "enabled", "effort": self.effort})
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        last = None
        for attempt in range(8):
            try:
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=180) as r:
                    out = json.loads(r.read())
                self.wall += time.perf_counter() - t0
                return out
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (401, 403):  # auth: do not retry
                    return None
                time.sleep([4, 10, 20, 40, 60, 90, 120, 180][attempt])
            except Exception:  # noqa: BLE001 - network: retry w/ backoff
                time.sleep([4, 10, 20, 40, 60, 90, 120, 180][attempt])
        return None

    def policy(self, window_alerts, summary=None):
        from .policies import stream_summary as _ss
        summary = summary if summary is not None else _ss(window_alerts)
        out = self._raw_call(summary)
        u = out["usage"]
        self.calls += 1
        self.prompt_tokens += u["prompt_tokens"]
        self.completion_tokens += u["completion_tokens"]
        if out is None:
            self.invalid += 1
            self.fallbacks += 1
            self.trace.append(dict(dims=list(DEFAULT_STATIC_POLICY["dims"]),
                                   window_min=DEFAULT_STATIC_POLICY["window_min"],
                                   source="api-failure"))
            return Policy.from_json(json.dumps(DEFAULT_STATIC_POLICY),
                                    "llm-fallback")
        content = (out["choices"][0]["message"].get("content") or "")
        import re
        m = re.search(r"\{[^{}]*\}", content, re.S)
        pol = Policy.from_json(m.group(0) if m else content, "llm-glm")
        if not pol.valid:
            self.invalid += 1
            self.fallbacks += 1
            pol = Policy.from_json(json.dumps(DEFAULT_STATIC_POLICY),
                                   "llm-fallback")
        self.trace.append(dict(dims=list(pol.dims),
                               window_min=pol.window_min, source=pol.source))
        return pol

    def policies_parallel(self, windows, workers=6, summaries=None):
        """Precompute per-window policies concurrently (stateless prompts),
        preserving order; returns list of Policy + fills counters.
        summaries: optional per-window summary blocks (dict or str),
        e.g. reformatted variants from the mechanism experiments; when
        given, they replace stream_summary(level=...) wholesale."""
        sums = summaries if summaries is not None else [
            stream_summary(w, level=self.summary_level) for w in windows]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            outs = list(ex.map(lambda s: self._raw_call(s), sums))
        pols = []
        import re
        for out in outs:
            if out is None:
                self.invalid += 1
                self.fallbacks += 1
                pol = Policy.from_json(json.dumps(DEFAULT_STATIC_POLICY),
                                       "llm-fallback")
                self.trace.append(dict(dims=list(pol.dims),
                                       window_min=pol.window_min,
                                       source="api-failure"))
                pols.append(pol)
                continue
            u = out["usage"]
            self.calls += 1
            self.prompt_tokens += u["prompt_tokens"]
            self.completion_tokens += u["completion_tokens"]
            content = (out["choices"][0]["message"].get("content") or "")
            m = re.search(r"\{[^{}]*\}", content, re.S)
            pol = Policy.from_json(m.group(0) if m else content, "llm-glm")
            if not pol.valid:
                self.invalid += 1
                self.fallbacks += 1
                pol = Policy.from_json(json.dumps(DEFAULT_STATIC_POLICY),
                                       "llm-fallback")
            self.trace.append(dict(dims=list(pol.dims),
                                   window_min=pol.window_min,
                                   source=pol.source))
            pols.append(pol)
        return pols


_LLM_USER = """Return a JSON grouping policy for the current 300-second window of IDS alerts.
Fields: "dims" = a list of ONE or TWO dimensions from ["src","dst","dst_port","family","sid"];
"window_min" = integer 1..120 (case inactivity timeout).
Choose the ONE or TWO dimensions whose equal values best indicate the same incident in THIS window.
Group alerts that plausibly belong to one incident; split distinct activities sharing an asset.
Example output: {{"dims": ["family", "dst"], "window_min": 10}}

Window summary:
{summary}

Respond with ONLY the JSON object now."""


def run_hybrid_precomputed(alerts, policies, delta_sec=300):
    """Execute a precomputed per-window policy sequence deterministically."""
    from .policies import DeterministicExecutor
    t0 = time.perf_counter()
    ex = DeterministicExecutor()
    from .arms import _windowize
    for w, pol in zip(_windowize(alerts, delta_sec), policies):
        ex.apply_window(w, pol)
    fb = sum(1 for p in policies if p.source.endswith("fallback"))
    return dict(cases=ex.case_partition(), wall=time.perf_counter() - t0,
                policy_calls=len(policies), fallbacks=fb,
                policy_trace=[dict(dims=list(p.dims),
                                   window_min=p.window_min,
                                   source=p.source) for p in policies])
