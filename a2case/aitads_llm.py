"""AIT-ADS LLM 0.5B arm: per-window policy generation (deployed generator).

The paper's deployed configuration is Qwen2.5-0.5B-Instruct; the CIC scale
ablation showed CPU-float32 and GPU-float16 produce identical partitions on
every scenario, so this rerun uses GPU-float16 for wall-clock feasibility
(disclosed in the run log). Per-scenario JSONs are written as completed so
the run is resumable; completed scenarios are skipped on re-run.

Run:  python -m a2i.aitads_llm [outdir]
"""
import json
import os
import sys

from . import aitads, arms
from .locals import LocalLLMPolicyGen
from . import metrics

QWEN = os.environ.get(
    "A2I_QWEN05",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "models",
                 "Qwen2.5-0.5B-Instruct"))


def main(outdir="results/gate0/llm05"):
    os.makedirs(outdir, exist_ok=True)
    gen = None
    for scen in aitads.SCENARIOS:
        out = os.path.join(outdir, f"{scen}.json")
        if os.path.exists(out):
            print(f"{scen}: already done", flush=True)
            continue
        alerts, incidents = aitads.load_stream(scen)
        n, n_inc = len(alerts), len(incidents)
        if gen is None:
            import torch
            gen = LocalLLMPolicyGen(
                os.path.abspath(QWEN), device="cuda",
                dtype=torch.float16)
        t0 = __import__("time").perf_counter()
        res = arms.run_hybrid(alerts, generator=gen)
        ev = metrics.evaluate(res["cases"], n, n_inc)
        rec = dict(scenario=scen, arm="hybrid-llm-0.5B-fp16-cuda",
                   dimset="per-window", n_alerts=n,
                   n_attack=sum(a.is_attack for a in alerts),
                   n_incidents=n_inc, **ev,
                   policy_calls=res["policy_calls"],
                   prompt_tokens=res["prompt_tokens"],
                   completion_tokens=res["completion_tokens"],
                   fallbacks=res["fallbacks"],
                   invalid=getattr(gen, "invalid", 0),
                   wall_llm=round(gen.wall, 1),
                   wall_total=round(__import__("time").perf_counter() - t0, 1),
                   policy_trace=res["policy_trace"])
        with open(out, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1)
        print(f"{scen}: ari={ev['ari']:.3f} calls={res['policy_calls']} "
              f"fb={res['fallbacks']} tok={res['prompt_tokens']}+"
              f"{res['completion_tokens']} "
              f"wall={rec['wall_total']}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/gate0/llm05")
