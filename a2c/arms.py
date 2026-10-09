"""The three L1 alert-to-case grouping arms.

All arms consume the identical windowed alert stream and produce a partition
of alerts into cases; they differ only in the grouping mechanism:

  static     fixed policy (src,dst; 10 min) — the incumbent SOAR baseline
  embedding  corpus-trained message embeddings + incremental cosine clustering
  hybrid     per-window policy generation (PSH stand-in or LLM) executed by
             the same deterministic engine as the static arm
"""

import json
import time

import numpy as np

from .policies import (DEFAULT_STATIC_POLICY, DeterministicExecutor, Policy,
                       LLMPolicyGen, psh_policy, stream_summary)


def _windowize(alerts, delta_sec):
    """Slice the sorted alert stream into micro-batch windows."""
    windows = []
    cur, start = [], (alerts[0].ts if alerts else 0.0)
    for a in alerts:
        if a.ts - start >= delta_sec:
            windows.append(cur)
            cur, start = [], a.ts
        cur.append(a)
    if cur:
        windows.append(cur)
    return windows


def run_static(alerts, delta_sec=300, dims=None, window_min=10):
    t0 = time.perf_counter()
    policy = Policy(dims=dims or DEFAULT_STATIC_POLICY["dims"],
                    window_min=window_min, source="static")
    ex = DeterministicExecutor()
    for w in _windowize(alerts, delta_sec):
        ex.apply_window(w, policy)
    return dict(cases=ex.case_partition(), wall=time.perf_counter() - t0,
                policy_calls=0, prompt_tokens=0, completion_tokens=0,
                fallbacks=0, policy_trace=[])


def run_hybrid(alerts, delta_sec=300, generator=None):
    """generator: None -> PSH stand-in; LLMPolicyGen -> real LLM policy calls."""
    t0 = time.perf_counter()
    ex = DeterministicExecutor()
    trace = []
    calls = ptok = ctok = fb = 0
    for w in _windowize(alerts, delta_sec):
        summ = stream_summary(w)
        if generator is None:
            pol = psh_policy(w, summ)
        else:
            pol = generator.policy(w, summ)
        ex.apply_window(w, pol)
        if generator is not None:
            calls += 1
            ptok += getattr(generator, "prompt_tokens", 0)
            ctok += getattr(generator, "completion_tokens", 0)
            fb += 1 if pol.source.endswith("fallback") else 0
        trace.append(dict(n=len(w), dims=list(pol.dims),
                          window_min=pol.window_min, source=pol.source))
    return dict(cases=ex.case_partition(), wall=time.perf_counter() - t0,
                policy_calls=calls, prompt_tokens=ptok, completion_tokens=ctok,
                fallbacks=fb, policy_trace=trace)


class _Embedder:
    """Corpus-trained average word embeddings (gensim Word2Vec), dim 64.

    Chosen over a pretrained transformer encoder for pilot reproducibility:
    it trains in seconds on the alert corpus itself and needs no model
    download; the arm's semantics (message-space similarity) are unchanged.
    The lexical skeleton of the templates (family keywords, entity
    addresses, ports) dominates the space, which is the intended behavior of
    a pure-semantic arm with no policy layer.
    """

    def __init__(self, texts, dim=64, seed=17):
        from gensim.models import Word2Vec
        sents = [self._tok(t) for t in texts]
        self.model = Word2Vec(sents, vector_size=dim, window=4, min_count=1,
                              workers=1, seed=seed, epochs=5)
        self.dim = dim

    @staticmethod
    def _tok(t):
        return [w.strip("[]:,") for w in t.lower().split() if w.strip("[]:,")]

    def embed(self, text):
        toks = self._tok(text)
        vecs = [self.model.wv[t] for t in toks if t in self.model.wv]
        if not vecs:
            return np.zeros(self.dim)
        return np.mean(vecs, axis=0)


def run_embedding(alerts, delta_sec=300, tau=0.75, case_window_min=30, seed=17,
                  encoder=None):
    """Incremental cosine clustering over message embeddings.

    An alert joins the most similar open case if cosine similarity >= tau
    (case open within case_window_min); otherwise it opens a new case. This
    is the pure-semantic extreme of the design space: no rules, no policy.
    encoder: None -> corpus-trained Word2Vec average embeddings; a
    MiniLMEmbedder instance -> pretrained sentence embeddings (reported as
    the fair semantic baseline). Similarity against open cases is
    vectorized; cases past the inactivity window are pruned.
    """
    t0 = time.perf_counter()
    if encoder is None:
        emb = _Embedder([a.text for a in alerts], seed=seed)
        texts_vecs = [emb.embed(a.text) for a in alerts]
    else:
        texts_vecs = encoder.embed_all([a.text for a in alerts])

    cases = []                    # dicts: alerts, sum (np), last_ts
    closed = []                   # finished cases (alerts only)
    last_prune = 0.0
    for a, v in zip(alerts, texts_vecs):
        # prune expired cases once per micro-batch of time
        if a.ts - last_prune >= delta_sec:
            last_prune = a.ts
            keep = []
            for c in cases:
                (closed if a.ts - c["last_ts"] > case_window_min * 60
                 else keep).append(c)
            cases = keep
        if not cases:
            best, best_sim = None, tau
        else:
            sums = np.stack([c["sum"] for c in cases])
            norms = np.linalg.norm(sums, axis=1)
            norms[norms == 0] = 1e-9
            nv = np.linalg.norm(v) or 1e-9
            sims = (sums @ v) / (norms * nv)
            j = int(np.argmax(sims))
            if sims[j] >= tau:
                best, best_sim = cases[j], float(sims[j])
            else:
                best, best_sim = None, tau
        if best is None:
            best = dict(alerts=[], sum=v.copy(), last_ts=a.ts)
            cases.append(best)
        best["alerts"].append(a)
        best["sum"] = best["sum"] + v
        best["last_ts"] = max(best["last_ts"], a.ts)
    partition = ([c["alerts"] for c in cases] +
                 [c["alerts"] for c in closed])
    partition = [c for c in partition if c]
    return dict(cases=partition, wall=time.perf_counter() - t0,
                policy_calls=0, prompt_tokens=0, completion_tokens=0,
                fallbacks=0, policy_trace=[])
