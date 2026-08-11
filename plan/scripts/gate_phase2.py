"""Phase 2 gate: generate real text through ROCM_FLASHINFER, and compare
greedy output against ROCM_AITER_FA on the same prompts.

Usage: python3 gate_phase2.py [MODEL] [BACKEND ...]
"""

import os
import sys

os.environ.setdefault("VLLM_ROCM_USE_FLASHINFER", "1")

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
BACKENDS = sys.argv[2:] or ["ROCM_FLASHINFER", "ROCM_AITER_FA"]

PROMPTS = [
    "The capital of France is",
    "Write one sentence about the ocean:",
    # long enough to span several 16-token pages and exercise chunked prefill
    (
        "Count from one to twenty in words: one, two, three, four, five, six, "
        "seven, eight, nine, ten, eleven, twelve, thirteen, fourteen, fifteen, "
        "sixteen, seventeen, eighteen, nineteen,"
    ),
]


def run(backend):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        attention_backend=backend,
        enforce_eager=True,
        max_model_len=2048,
        # This box is shared; another workload holds most of the VRAM.
        gpu_memory_utilization=float(os.environ.get("GPU_MEM_UTIL", "0.08")),
        dtype="bfloat16",
        disable_log_stats=True,
    )
    # Greedy so the two backends are directly comparable.
    sp = SamplingParams(temperature=0.0, max_tokens=24, logprobs=5)
    outs = llm.generate(PROMPTS, sp)
    result = []
    for o in outs:
        c = o.outputs[0]
        # Per-step top-k logprobs, so a divergence can be classified as a
        # near-tie rather than just reported as a mismatch.
        steps = []
        for lp in c.logprobs or []:
            ranked = sorted(lp.items(), key=lambda kv: -kv[1].logprob)
            steps.append([(tid, v.logprob, v.decoded_token) for tid, v in ranked])
        result.append({"text": c.text, "ids": list(c.token_ids), "steps": steps})
    return result


def main():
    results = {}
    for be in BACKENDS:
        print(f"\n{'=' * 70}\n=== {be}\n{'=' * 70}", flush=True)
        try:
            results[be] = run(be)
            for i, r in enumerate(results[be]):
                print(f"  [{i}] {r['text']!r}")
        except Exception:
            import traceback

            traceback.print_exc()
            print(f"=== {be} FAILED")
            results[be] = None

    print(f"\n{'=' * 70}\n=== VERDICT\n{'=' * 70}")
    for be, r in results.items():
        print(f"  {be:<20} {'ran' if r else 'FAILED'}")

    if len(BACKENDS) == 2 and all(results.get(b) for b in BACKENDS):
        a, b = BACKENDS
        same = 0
        for i, (ra, rb) in enumerate(zip(results[a], results[b])):
            match = ra["ids"] == rb["ids"]
            same += match
            print(f"  prompt[{i}] greedy token ids match: {match}")
            if not match:
                # Where do they diverge, and was it a near-tie?
                n = min(len(ra["ids"]), len(rb["ids"]))
                d = next((k for k in range(n) if ra["ids"][k] != rb["ids"][k]), n)
                print(f"      diverges at token {d}")
                print(f"      {a}: {ra['text']!r}")
                print(f"      {b}: {rb['text']!r}")
                for name, r in ((a, ra), (b, rb)):
                    if d < len(r["steps"]) and len(r["steps"][d]) >= 2:
                        top = r["steps"][d]
                        gap = top[0][1] - top[1][1]
                        print(
                            f"      {name} step {d}: top1={top[0][2]!r} "
                            f"({top[0][1]:.6f})  top2={top[1][2]!r} "
                            f"({top[1][1]:.6f})  gap={gap:.2e}"
                        )
        print(f"\n  {same}/{len(PROMPTS)} prompts identical")


if __name__ == "__main__":
    main()
