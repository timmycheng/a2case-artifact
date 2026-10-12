"""Local LLM policy generator and pretrained-embedding variant.

LocalLLMPolicyGen: a real LLM (small open-weights model, CPU inference)
behind the exact policy contract of policies.LLMPolicyGen — same prompt,
same JSON validation, same static fallback, real token counts.

MiniLMEmbedder: pretrained sentence encoder for the embedding arm
(drop-in for the corpus-trained Word2Vec embedder; the paper reports both).
"""

import json
import re
import time

import numpy as np

from .policies import Policy, DEFAULT_STATIC_POLICY

_SYSTEM = ("You are a configuration agent for a security-alert pipeline. "
           "You always respond with ONLY a single JSON object and nothing else.")

_LLM_USER = """Return a JSON grouping policy for the current 300-second window of IDS alerts.
Fields: "dims" = a list of ONE or TWO dimensions from ["src","dst","dst_port","family","sid"];
"window_min" = integer 1..120 (case inactivity timeout).
Choose the ONE or TWO dimensions whose equal values best indicate the same incident in THIS window.
Group alerts that plausibly belong to one incident; split distinct activities sharing an asset.
Example output: {{"dims": ["family", "dst"], "window_min": 10}}

Window summary:
{summary}

Respond with ONLY the JSON object now."""


class LocalLLMPolicyGen:
    """Real LLM policy generation on CPU (open-weights, pinned revision)."""

    def __init__(self, model_path, max_new_tokens=96, seed=17,
                 device=None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(seed)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        # defaults reproduce the paper's deployed configuration exactly:
        # float32 weights on CPU. The scale ablation passes explicit
        # device/dtype for the larger family members (disclosed in-paper).
        device = device or "cpu"
        dtype = dtype or torch.float32
        if device == "auto":
            # hybrid GPU/CPU placement for weights larger than VRAM
            # (accelerate device_map); identical weights/precision, only
            # the deployment differs -- disclosed in the paper
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype, device_map="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.fallbacks = 0
        self.invalid = 0
        self.wall = 0.0
        self.trace = []

    def policy(self, window_alerts, summary=None):
        from .policies import stream_summary
        summary = summary or stream_summary(window_alerts)
        block = (json.dumps(summary, indent=1)
                 if isinstance(summary, dict) else summary)
        user = _LLM_USER.format(summary=block)
        return self.generate_policy_text(user, n_window=len(window_alerts))

    def generate_policy_text(self, user_text: str, n_window: int = 0):
        """Chat-template + greedy generate + JSON-policy parse for an
        already-formatted user prompt. ``policy`` builds the standard
        prompt; mechanism experiments (s4_reversion) reformat only the
        summary block and call this entry so decoding, validation,
        fallback and token accounting stay byte-identical."""
        ids = self.tok.apply_chat_template(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": user_text}],
            add_generation_prompt=True, return_tensors="pt")
        ids = ids.to(self.model.device)
        n_in = ids.shape[1]
        t0 = time.perf_counter()
        with self.torch.no_grad():
            out = self.model.generate(
                ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tok.eos_token_id)
        self.wall += time.perf_counter() - t0
        n_out = out.shape[1] - n_in
        self.calls += 1
        self.prompt_tokens += n_in
        self.completion_tokens += n_out
        content = self.tok.decode(out[0, n_in:], skip_special_tokens=True)
        m = re.search(r"\{[^{}]*\}", content, re.S)
        pol = Policy.from_json(m.group(0) if m else content, "llm")
        if not pol.valid:
            self.invalid += 1
            self.fallbacks += 1
            pol = Policy.from_json(json.dumps(DEFAULT_STATIC_POLICY),
                                   "llm-fallback")
        self.trace.append(dict(n=n_window, dims=list(pol.dims),
                               window_min=pol.window_min, source=pol.source,
                               raw=content[:120]))
        return pol


class MiniLMEmbedder:
    """Pretrained sentence encoder (all-MiniLM-L6-v2), mean pooled, 384-d."""

    def __init__(self, model_path):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path)

    def embed_all(self, texts, batch_size=256):
        vecs = self.model.encode(texts, batch_size=batch_size,
                                 show_progress_bar=False,
                                 normalize_embeddings=True)
        return [np.asarray(v, dtype=np.float64) for v in vecs]
