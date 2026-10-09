"""Grouping policies: schema, validation, deterministic executor.

The A2I architecture position: a policy generator emits a grouping policy per
micro-batch window; a deterministic engine executes it. In the full system the
generator is an LLM whose output is schema-validated (invalid output falls
back to the static policy). For the pilot runs in this study the generator is
a deterministic policy-induction procedure (PSH, policy-synthesis heuristic)
that consumes exactly the same compact stream summary and emits exactly the
same validated JSON contract, so the architecture position — policy generated
per window at run time, executed deterministically — is exercised end to end
without an API dependency. The LLM generator is a drop-in via LLMPolicyGen.
"""

import json
import os
import re
import urllib.request
from dataclasses import dataclass

VALID_DIMS = {"src", "dst", "dst_port", "family", "sid", "src_dst", "asset"}

DEFAULT_STATIC_POLICY = {
    "dims": ["src", "dst"],
    "window_min": 10,
    "labels": {},
    "edge_rules": [],
}


@dataclass
class Policy:
    dims: list
    window_min: int
    labels: dict = None
    edge_rules: list = None
    source: str = "static"          # static | psh | llm
    valid: bool = True
    raw: str = ""

    @classmethod
    def from_json(cls, s: str, source: str):
        try:
            obj = json.loads(s)
            dims = obj["dims"]
            if not isinstance(dims, list) or not dims:
                raise ValueError("dims must be a non-empty list")
            if any(d not in VALID_DIMS for d in dims):
                raise ValueError(f"unknown dim in {dims}")
            w = int(obj.get("window_min", 10))
            if not 1 <= w <= 120:
                raise ValueError("window_min out of range")
            return cls(dims=dims, window_min=w,
                       labels=obj.get("labels", {}) or {},
                       edge_rules=obj.get("edge_rules", []) or [],
                       source=source, valid=True, raw=s)
        except Exception:
            return cls(dims=DEFAULT_STATIC_POLICY["dims"],
                       window_min=DEFAULT_STATIC_POLICY["window_min"],
                       source=source, valid=False, raw=s)


def policy_key(alert, dims):
    """Deterministic grouping key of an alert under a policy."""
    parts = []
    for d in dims:
        if d == "src":
            parts.append(alert.src)
        elif d == "dst":
            parts.append(alert.dst)
        elif d == "dst_port":
            parts.append(str(alert.dst_port))
        elif d == "family":
            parts.append(alert.family)
        elif d == "sid":
            parts.append(alert.sid)
        elif d == "src_dst":
            parts.append(f"{alert.src}>{alert.dst}")
        elif d == "asset":
            parts.append(alert.dst)     # victim-side asset identity
    return "|".join(parts)


class DeterministicExecutor:
    """Executes any policy over a windowed stream.

    Case semantics: an alert joins the open case with the same aggregation
    key if it arrives within the policy's inactivity timeout of that case's
    last alert, else it opens a new case (the previously open case with that
    key is closed; closed cases are never re-opened). At most one case is
    open per key at any time, so the executor is O(alerts).
    """

    def __init__(self):
        self.open_by_key = {}    # key -> case dict
        self.closed = []
        self._n = 0

    def apply_window(self, alerts, policy: Policy):
        w_sec = policy.window_min * 60
        for a in alerts:
            key = policy_key(a, policy.dims)
            c = self.open_by_key.get(key)
            if c is not None and a.ts - c["last_ts"] <= w_sec:
                c["alerts"].append(a)
                if a.ts > c["last_ts"]:
                    c["last_ts"] = a.ts
            else:
                if c is not None:
                    self.closed.append(c)
                self._n += 1
                c = dict(key=key, alerts=[a], last_ts=a.ts, first_ts=a.ts,
                         policy=policy.source)
                self.open_by_key[key] = c
        return self

    def close_all(self):
        for c in self.open_by_key.values():
            self.closed.append(c)
        self.open_by_key.clear()
        return self

    def snapshot(self):
        """Capture executor state for trial-and-restore (oracle search).
        Alert lists are copied so trials never mutate the recorded state."""
        snap = {k: dict(c, alerts=list(c["alerts"]))
                for k, c in self.open_by_key.items()}
        return snap, len(self.closed)

    def restore(self, snap):
        snap_dict, n_closed = snap
        self.open_by_key = {k: dict(c, alerts=list(c["alerts"]))
                            for k, c in snap_dict.items()}
        del self.closed[n_closed:]

    def case_partition(self):
        self.close_all()
        return [c["alerts"] for c in self.closed if c["alerts"]]


# ---------------------------------------------------------------------------
# Policy generators
# ---------------------------------------------------------------------------

def stream_summary(window_alerts):
    """Compact summary of a micro-batch window (the generator's only input)."""
    fam_count, dst_count, pair_count = {}, {}, {}
    for a in window_alerts:
        fam_count[a.family] = fam_count.get(a.family, 0) + 1
        dst_count[a.dst] = dst_count.get(a.dst, 0) + 1
        pair = f"{a.src}>{a.dst}"
        pair_count[pair] = pair_count.get(pair, 0) + 1
    return dict(
        n=len(window_alerts),
        families=sorted(fam_count.items(), key=lambda kv: -kv[1])[:8],
        top_dsts=sorted(dst_count.items(), key=lambda kv: -kv[1])[:6],
        top_pairs=sorted(pair_count.items(), key=lambda kv: -kv[1])[:6],
    )


def psh_policy(window_alerts, summary=None):
    """Policy-synthesis heuristic (deterministic stand-in for the LLM).

    Induces a grouping policy from the window summary alone, by the same
    decision an analyst would encode: keep the victim-side service identity
    (dst + port) together when one or few destinations dominate the window;
    split by signature family when several families crowd one destination
    (mixed-incident regime); fall back to src+dst when activity is diffuse.

    Emits the same JSON contract the LLM must return; validation and
    execution are identical for both generators.
    """
    summary = summary or stream_summary(window_alerts)
    n = summary["n"]
    if n == 0:
        return Policy.from_json(json.dumps(DEFAULT_STATIC_POLICY), "psh")
    top_dst, top_dst_n = summary["top_dsts"][0] if summary["top_dsts"] else ("", 0)
    n_fam = len(summary["families"])
    # single dominant destination + few families -> service-keyed grouping
    if top_dst_n >= 0.5 * n and n_fam <= 3:
        dims = ["dst", "dst_port"]
        w = 10
    # crowded window, many families on shared infrastructure -> family split
    elif n_fam >= 4 and top_dst_n >= 0.3 * n:
        dims = ["family", "dst"]
        w = 5
    else:
        dims = ["src", "dst"]
        w = 10
    return Policy.from_json(json.dumps(dict(dims=dims, window_min=w)), "psh")


_PROMPT = """You are configuring the grouping layer of a security-alert pipeline.
Given this summary of the current {delta_sec}-second window of IDS alerts, return
a JSON grouping policy with fields:
  "dims": non-empty subset of ["src","dst","dst_port","family","sid"],
  "window_min": integer 1..120 (case inactivity timeout).
Rules: group alerts that plausibly belong to one incident; split distinct
activities sharing an asset. Return ONLY the JSON object.

Window summary:
{summary}
"""


class LLMPolicyGen:
    """Drop-in LLM policy generator (OpenAI-compatible chat completions).

    Enabled by env: A2C_LLM_BASE_URL, A2C_LLM_API_KEY, A2C_LLM_MODEL.
    Records prompt/completion tokens per call for the cost model; invalid
    JSON falls back to the static policy (fallback rate is reported).
    """

    def __init__(self, delta_sec=300):
        self.base_url = os.environ["A2C_LLM_BASE_URL"].rstrip("/")
        self.api_key = os.environ.get("A2C_LLM_API_KEY", "")
        self.model = os.environ.get("A2C_LLM_MODEL", "gpt-4o-mini")
        self.delta_sec = delta_sec
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.fallbacks = 0

    def policy(self, window_alerts, summary=None):
        summary = summary or stream_summary(window_alerts)
        prompt = _PROMPT.format(delta_sec=self.delta_sec,
                                summary=json.dumps(summary, indent=1))
        body = json.dumps(dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=120,
        )).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            self.calls += 1
            u = out.get("usage", {})
            self.prompt_tokens += u.get("prompt_tokens", 0)
            self.completion_tokens += u.get("completion_tokens", 0)
            content = out["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)
            pol = Policy.from_json(m.group(0) if m else content, "llm")
            if not pol.valid:
                self.fallbacks += 1
                pol = Policy.from_json(json.dumps(DEFAULT_STATIC_POLICY), "llm-fallback")
            return pol
        except Exception:
            self.fallbacks += 1
            return Policy.from_json(json.dumps(DEFAULT_STATIC_POLICY), "llm-fallback")
