Title: [ROCm] Detach EngineCore roots before final collection
Proposed branch: `akaratza_cuda_teardown_enginecore`
Depends on: `akaratza_cuda_teardown_multiproc`

# Description

`EngineCore.shutdown()` currently performs its full collection while the Core local, signal callback, I/O threads, executor, and scheduler can still retain the object graph. A zero request-drain timeout can also leave no independent budget for native cleanup: in two earlier TP=1 MI300X diagnostics the frontend internally force-killed EngineCore even though the outer harness observed a zero API-server exit. This change separates request draining from a named 15-second resource-cleanup grace—leaving margin above the 10.939-second maximum in the two local TP=2 candidate runs—stops Core-owned callbacks and socket threads, releases executor and scheduler roots after worker teardown, and runs one measured final collection while Python, PyTorch, HIP, and logging remain usable.

- Bound and join the signal callback and EngineCore I/O threads before dependent state is released.
- Coalesce repeated terminal signals through native cleanup, restore default handlers, sever closure and component roots, and then collect.
- Preserve incomplete graph or frontend state for the exit-only ROCm freeze and prevent cleanup failures from becoming a reported clean exit.
- Keep Ray on its existing actor contract and always release MPClient frontend resources after the process manager finishes.
- Validate timeout-zero FULL graphs twice on TP=2 MI300X, with the second run reusing the same GPUs and port: both requests returned HTTP 200; each run reset eight graphs (four per worker, 1.73-2.00 ms reset-marker spans, zero reset failures); the executor logged graceful worker exits without worker escalation; the API server exited zero without harness SIGKILL; the process group was empty; and final VRAM samples were back at the 286 MiB-per-GPU baseline.

This PR does not make EngineCore construction transactional, remove the modern-ROCm freeze, patch PyTorch, cover Ray teardown, or claim MI355/100-cycle soak completion. The API layer's pre-existing output-handler `EngineDeadError` log after an otherwise clean shutdown is also left for a separate change.

Local evidence uses interrupted-cleanup run IDs `7bf617b7` and `aa572ed8`; their `child.stdout.log` records, rather than `forced_signal`, show the internal EngineCore force-kill. Final TP=2 run IDs `e4f829f5` and `92c97ec8` measured signal-to-API-reap at 10.939 s and 10.151 s, last-worker-shutdown-marker-to-reap at 9.766 s and 9.008 s, signal-to-root-detach at 7.416 s and 7.041 s, and final collection at 0.877 s and 0.453 s. Exact 286 MiB baselines were first re-observed 7.042 s and 7.240 s after signal; the two-second VRAM-stability checks completed 13.669 s and 12.465 s after signal (2.730 s and 2.314 s after API reap). These uncommitted samples validate cleanup order and the defensive bound; they do not attribute the roughly ten-second process tail to graph reset or establish percentile performance. Attach the raw artifacts somewhere reviewers can access before posting.

Reproduce the final scenario twice; the harness creates a unique run directory each time, and the second invocation exercises the same GPU/port restart:

```bash
env -u VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS HIP_VISIBLE_DEVICES=1,2 \
  .venv/bin/python benchmarks/shutdown_timing.py \
  --output-dir /tmp/vllm-shutdown-evidence \
  --scenario completed --cwd "$PWD" --gpu 1,2 \
  --health-url http://127.0.0.1:18131/health \
  --request-url http://127.0.0.1:18131/v1/completions \
  --model hmellor/tiny-random-LlamaForCausalLM --max-tokens 8 \
  --startup-timeout 300 --request-timeout 120 --shutdown-timeout 120 \
  --force-kill-timeout 30 --vram-timeout 90 --vram-stable-seconds 2 \
  --vram-allowance-mib 512 --poll-interval 0.1 -- \
  .venv/bin/python -m vllm.entrypoints.cli.main serve \
  hmellor/tiny-random-LlamaForCausalLM --host 127.0.0.1 --port 18131 \
  --max-model-len 128 --max-num-seqs 8 --gpu-memory-utilization 0.2 \
  --tensor-parallel-size 2 --distributed-executor-backend mp \
  --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}' \
  --shutdown-timeout 0
```
