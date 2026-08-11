"""Phase 3 layer 3: GSM8K accuracy, ROCM_FLASHINFER vs a reference backend.

The packaged test (tests/evals/gsm8k/test_gsm8k_correctness.py) parametrizes
over quantized models from configs/ and offers no way to pick an attention
backend, so drive the same `evaluate_gsm8k` it uses against servers we launch
ourselves, one per backend.

Layers 1 and 2 of Phase 3 (SDPA reference, greedy token match) check that the
kernels compute correct attention and that two backends agree on a handful of
short prompts. Neither would catch a defect that only shows up as degraded
reasoning quality over hundreds of multi-step problems. That is what this is
for.

Usage: python3 gate_phase3_gsm8k.py [MODEL] [BACKEND ...]
"""

import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

os.environ.setdefault("VLLM_ROCM_USE_FLASHINFER", "1")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))


from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Mistral-7B-Instruct-v0.3"
BACKENDS = sys.argv[2:] or ["ROCM_FLASHINFER", "ROCM_AITER_FA"]

SERVED_NAME = "gsm8k-model"
PORT = int(os.environ.get("PORT", "8137"))
NUM_QUESTIONS = int(os.environ.get("NUM_QUESTIONS", "400"))
NUM_SHOTS = int(os.environ.get("NUM_SHOTS", "5"))
GPU_MEM_UTIL = os.environ.get("GPU_MEM_UTIL", "0.85")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "4096")
# The plan's gate: the two backends must land within this of each other.
RTOL = 0.08
STARTUP_TIMEOUT = 900


def wait_healthy(port: int, proc: subprocess.Popen, timeout: int) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"  server exited early with code {proc.returncode}")
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(5)
    return False


def run_backend(backend: str) -> dict | None:
    # Default to /tmp, but allow a mounted directory so the server log survives
    # a --rm container when the launch itself is what failed.
    log_dir = os.environ.get("LOG_DIR", "/tmp")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"gsm8k_server_{backend}.log")
    cmd = [
        "vllm",
        "serve",
        MODEL,
        "--served-model-name",
        SERVED_NAME,
        "--attention-backend",
        backend,
        "--port",
        str(PORT),
        "--max-model-len",
        MAX_MODEL_LEN,
        "--gpu-memory-utilization",
        GPU_MEM_UTIL,
        "--dtype",
        "bfloat16",
    ]
    print(f"  launching: {' '.join(cmd)}", flush=True)
    env = dict(os.environ, VLLM_ROCM_USE_FLASHINFER="1")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_healthy(PORT, proc, STARTUP_TIMEOUT):
            print(f"  server never became healthy; see {log_path}")
            return None
        print("  server up, running GSM8K", flush=True)
        t0 = time.time()
        results = evaluate_gsm8k(
            num_questions=NUM_QUESTIONS,
            num_shots=NUM_SHOTS,
            model=SERVED_NAME,
            host="http://127.0.0.1",
            port=PORT,
            temperature=0.0,
            seed=42,
        )
        results["wall_s"] = time.time() - t0
        return results
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=60)
        # The port takes a moment to free before the next server can bind.
        time.sleep(15)


def main() -> int:
    print(f"model={MODEL}  questions={NUM_QUESTIONS}  shots={NUM_SHOTS}")
    results: dict[str, dict | None] = {}
    for backend in BACKENDS:
        print(f"\n{'=' * 70}\n=== {backend}\n{'=' * 70}", flush=True)
        try:
            results[backend] = run_backend(backend)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  {backend} FAILED: {e}")
            results[backend] = None
        r = results[backend]
        if r:
            print(
                f"  accuracy={r['accuracy']:.4f}  "
                f"invalid={r.get('invalid_rate', float('nan')):.4f}  "
                f"wall={r['wall_s']:.0f}s"
            )

    print(f"\n{'=' * 70}\n=== GSM8K SUMMARY\n{'=' * 70}")
    for backend, r in results.items():
        acc = f"{r['accuracy']:.4f}" if r else "FAILED"
        print(f"  {backend:<20} accuracy={acc}")

    ok = all(results.get(b) for b in BACKENDS)
    if ok and len(BACKENDS) == 2:
        a, b = BACKENDS
        delta = abs(results[a]["accuracy"] - results[b]["accuracy"])
        print(f"\n  |delta| = {delta:.4f}   gate: <= {RTOL}")
        verdict = "PASS" if delta <= RTOL else "FAIL"
        print(f"  PHASE 3 LAYER 3: {verdict}")
        return 0 if verdict == "PASS" else 1
    print("\n  PHASE 3 LAYER 3: INCONCLUSIVE (a backend failed to run)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
