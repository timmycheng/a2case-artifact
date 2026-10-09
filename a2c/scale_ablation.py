"""Same-family scale ablation: Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct
(stage-4; advisor next_step.md, addition 4).

Turns the paper's two-point scale contrast (0.5B instruction model vs. the
GLM-5.3 frontier reasoning model) into a controlled dose-response curve
within the Qwen2.5 instruction-tuned family: 0.5 -> 1.5 -> 3 -> 7 B, same
policy contract, same prompt, same greedy decoding, same grid (the seven
original scenarios incl. the D2 twin stress test). GLM-5.3 stays in the
paper as the training-objective contrast, not a scale endpoint.

Compute disclosure (single consumer GPU, RTX 4060 Ti 8 GB): members that
fit in VRAM run float16 on GPU; the 7B member does not fit and runs
float16 on CPU. The paper's deployed 0.5B configuration (float32, CPU) is
re-run in-run as the bridging anchor between the published numbers and the
curve conditions; the 0.5B fp16-GPU condition isolates the precision/
device effect from the size effect.

Entry point:
    python -m a2c.scale_ablation --data <csv-dir> \
        --out results/scale_ablation.json --models 0.5B,1.5B,3B,7B
"""

import argparse
import json
import os

import torch

from . import arms, metrics, scenarios
from .locals import LocalLLMPolicyGen
from .round3 import evaluate_all

MODEL_ROOT = os.environ.get("A2C_MODEL_ROOT", "./models")
QWEN_05 = r"..\data\models\Qwen2.5-0.5B-Instruct"

# (label, path, device, dtype) -- dtype "fp16" -> float16; device "gpu"
# requires CUDA. The 0.5B fp32-CPU anchor replicates the paper config.
CONDITIONS = [
    ("qwen0.5B_fp32cpu_anchor", QWEN_05, "cpu", "fp32"),
    ("qwen0.5B_fp16gpu", QWEN_05, "gpu", "fp16"),
    ("qwen1.5B_fp16gpu", os.path.join(MODEL_ROOT, "Qwen2.5-1.5B-Instruct"),
     "gpu", "fp16"),
    ("qwen3B_fp16gpu", os.path.join(MODEL_ROOT, "Qwen2.5-3B-Instruct"),
     "gpu", "fp16"),
    # exoneration anchor: 3B under the SAME hybrid deployment as 7B --
    # if it stays at the pure-GPU level, the 7B collapse is attributable
    # to the model, not to the offload kernel path
    ("qwen3B_fp16_offload", os.path.join(MODEL_ROOT, "Qwen2.5-3B-Instruct"),
     "auto", "fp16"),
    ("qwen7B_fp16_offload", os.path.join(MODEL_ROOT, "Qwen2.5-7B-Instruct"),
     "auto", "fp16"),
]

ORIGINAL = [("A1", "A1"), ("A2", "A2"), ("A3", "A3"), ("B2", "B2"),
            ("B3", "B3"), ("E", "E"), ("D2", "D2")]


def resolve(dtype):
    return torch.float16 if dtype == "fp16" else torch.float32


def make_generator(label, path, device, dtype, seed=17):
    dev = ("cuda" if device == "gpu" and torch.cuda.is_available()
           else device)
    if device == "gpu" and dev != "cuda":
        print(f"  [{label}] CUDA unavailable, falling back to cpu",
              flush=True)
    try:
        return LocalLLMPolicyGen(path, device=dev, dtype=resolve(dtype),
                                 seed=seed)
    except torch.cuda.OutOfMemoryError:
        print(f"  [{label}] GPU OOM, retrying on cpu/{dtype}", flush=True)
        return LocalLLMPolicyGen(path, device="cpu", dtype=resolve(dtype),
                                 seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/scale_ablation.json")
    ap.add_argument("--models", default=",".join(
        c[0] for c in CONDITIONS))
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    zf = args.data
    want = set(args.models.split(","))
    rows = []
    if os.path.exists(args.out):
        # resume: a previous --models batch may already have rows
        try:
            rows = json.load(open(args.out))
        except Exception:
            rows = []
    json.dump(rows, open(args.out, "w"), indent=1)

    builders = dict(ORIGINAL)
    for label, path, device, dtype in CONDITIONS:
        if label not in want:
            continue
        if "7B" in label and not os.path.isdir(path):
            print(f"[skip] {label}: model dir missing ({path})", flush=True)
            continue
        g = make_generator(label, path, device, dtype, seed=args.seed)
        dev = getattr(g, "device", "?")
        for name in builders:
            build = {
                "A1": lambda: scenarios.build_scenario(zf, "A1", seed=17),
                "A2": lambda: scenarios.build_scenario(zf, "A2", seed=17),
                "A3": lambda: scenarios.build_scenario(zf, "A3", seed=17),
                "B2": lambda: scenarios.build_scenario(zf, "B2", seed=17),
                "B3": lambda: scenarios.build_scenario(zf, "B3", seed=17),
                "E": lambda: scenarios.build_scenario(zf, "E", seed=17),
                "D2": lambda: scenarios.build_twins(zf, seed=17),
            }[name]
            sc = build()
            d = {}
            d["hybrid_llm"] = arms.run_hybrid(sc.alerts, generator=g)
            d["hybrid_llm"]["prompt_tokens"] = g.prompt_tokens
            d["hybrid_llm"]["completion_tokens"] = g.completion_tokens
            sc.stats["seed"] = args.seed
            before = len(rows)
            evaluate_all(name, sc, d, rows, "scale_ablation",
                         llm_meta=dict(model=label, device=dev, dtype=dtype,
                                       llm_valid=g.calls - g.fallbacks,
                                       llm_calls=g.calls))
            # evaluate_all appends one row per arm; tag them with the model
            for r in rows[before:]:
                r["model"] = label
                r["model_device"] = dev
                r["model_dtype"] = dtype
            json.dump(rows, open(args.out, "w"), indent=1)
        print(f"== {label} done: calls={g.calls} "
              f"fallbacks={g.fallbacks} wall={g.wall:.0f}s", flush=True)
        del g
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("scale ablation complete:", len(rows), "rows", flush=True)


if __name__ == "__main__":
    main()
