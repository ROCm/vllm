Title: [ROCm] Add CUDA graph teardown measurement tooling
Proposed branch: `akaratza_cuda_teardown_tooling`

# Description

Current PyTorch ROCm builds synchronize from each `CUDAGraph` destructor, but the call-count issue alone does not prove that it causes vLLM's shutdown tail. This change adds reproducible measurement tools before changing ownership policy: a standalone graph-lifetime benchmark for idle and in-flight destruction, an external service-lifecycle harness with parent-authoritative process and VRAM timing, and an opt-in `os.write` marker channel that remains a no-op unless enabled. On MI300/gfx942 with PyTorch `2.12.0+git6bbd260` and HIP `7.2.53211`, an `N=64` trace placed all 64 `hipGraphExecDestroy` calls in explicit reset (10.51 ms total), then recorded 64 additional `hipDeviceSynchronize` calls during wrapper destruction (0.039 ms total); this proves the destructor call pattern, not a production latency root cause.

- Add `del`/GC, explicit-reset, and in-flight-reset graph-lifetime modes with structured timing and memory output.
- Add scoped process launch, direct graceful service signaling, bounded process-group escalation, VRAM/RSS/FD/thread sampling, and marker validation.
- Register the optional marker descriptor, path, and run ID without affecting compilation hashes.
- Validate the tools with their self-tests, MI300 runs at multiple graph counts, `rocprofv3`, and `rocgdb`; the local trace identifier is `rocprof-final-quiesced-20260819/3283726_hip_api_trace.csv` and must be attached somewhere reviewers can access before posting.

This PR adds evidence and diagnostics only; it does not change vLLM graph teardown policy or claim that the PyTorch destructor is the production latency root cause.

Reproduce the quoted trace on one idle GPU (persist stdout and marker stderr beside the profiler output before attaching the artifact to the PR):

```bash
mkdir -p /tmp/rocprof-cudagraph-n64
HIP_VISIBLE_DEVICES=3 rocprofv3 --runtime-trace --stats --summary \
  --summary-output-file /tmp/rocprof-cudagraph-n64/summary.txt \
  -d /tmp/rocprof-cudagraph-n64 -f csv -- \
  .venv/bin/python benchmarks/cudagraph_lifetime.py \
  -n 64 --mode explicit-reset-one-sync --device cuda:0 \
  >result.json 2>phase-markers.jsonl
```
