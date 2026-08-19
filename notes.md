# ROCm CUDA graph teardown handoff

## Status and guardrails

This work investigates deterministic CUDA graph teardown on ROCm and makes the
vLLM shutdown dependency order explicit from graph owners through workers,
executors, and EngineCore. The implementation is currently a full-stack work in
progress, not a finished PR.

- Current branch: `akaratza_shutdown_wip` (intentional).
- Base: `d3fafe0c27f9666a06675858738aaeab949da0f5`, the current local
  `origin/main` at the time of the work.
- This document is part of one explicitly authorized WIP snapshot targeted at
  `rocm/akaratza_shutdown_wip`. It is not a merge-ready PR, and no PR has been
  opened.
- Seven committed `pr_*.md` drafts describe a proposed stacked split; they are
  planning documents, not existing pull requests.
- Before this handoff was added, the shared worktree had 29 modified tracked
  files and 13 untracked files. The tracked diff was 4,672 insertions and 308
  deletions. When splitting the WIP snapshot, stage each proposed PR by an
  explicit file list rather than copying the full branch.
- Raw logs, profiler output, and result JSON are local and uncommitted. Upload a
  reviewer-accessible, sanitized evidence bundle before opening PRs.

The original proposal was supplied outside the repository and is intentionally
not committed because it contains preliminary figures without retained raw
evidence. Its relevant scope is summarized here. The acceptance matrix is not
complete, and evidence below must take precedence over proposed targets or
unverified historical numbers.

## Candid result

There is **no measured CUDA graph shutdown speedup yet**. There is no controlled
pre-change versus post-change service benchmark, only two final full-stack
candidate trials, so no latency delta or percentile claim is justified.

The strongest current evidence is about correctness and observability:

- graph ownership is enumerated and reset in a deliberate dependency order;
- two TP=2 FULL-mode candidate shutdowns completed without worker escalation or
  harness SIGKILL, returned VRAM to baseline, and left no process group behind;
- earlier five-second/default controls interrupted workers, which motivates a
  bounded ROCm worker grace period; and
- EngineCore/root cleanup is now separately bounded and measured, which avoids
  coupling native resource cleanup to a zero request-drain timeout.

The four measured per-worker graph-reset marker spans were only 1.728946,
1.995571, 1.826917, and 1.941084 ms (four graphs each). They do **not** explain
the roughly 10-11 second signal-to-process-reap tail. The EngineCore work is
motivated by cleanup order, retained Python roots, and earlier interrupted
cleanup—not by a demonstrated graph-reset latency bottleneck.

## Change map

| Layer | Main files | Intent |
| --- | --- | --- |
| Measurement | `benchmarks/cudagraph_lifetime.py`, `benchmarks/shutdown_timing.py`, `vllm/utils/shutdown_markers.py`, `vllm/envs.py`, `vllm/v1/utils.py` | Standalone graph-lifetime tracing; process-group shutdown harness; opt-in, failure-aware phase markers. |
| Ownership core | `vllm/compilation/cuda_graph.py`, `vllm/compilation/breakable_cudagraph.py` | Owner protocol/registry; graph identity and device tracking; quiesce, enumerate, synchronize, reset, clear, retention, and failure policy; one common graph-construction path. |
| Graph owners | `vllm/v1/worker/gpu/cudagraph_utils.py`, both GPU model runners, encoder graph code, Gemma4, and base/concrete speculators | Register each graph-containing object and expose explicit teardown. |
| UBatch | `vllm/v1/worker/gpu_ubatch_wrapper.py`, `vllm/v1/worker/ubatching.py` | Stop admission/capture, coordinate the background thread/barrier, and retain unsafe state at terminal failure boundaries. |
| Runner/worker | both `gpu_model_runner.py` files, `vllm/v1/worker/gpu_worker.py`, `vllm/v1/executor/uniproc_executor.py` | Create the registry before graph construction; reset graphs before model/KV/pool state; keep ordinary non-ROCm shutdown direct; use staged ROCm terminal teardown. |
| Multiprocessing | `vllm/v1/executor/multiproc_executor.py` | Stop async output before worker graph teardown; release queues/distributed state in dependency order; report nonzero worker exits. |
| Engine/frontend | `vllm/v1/engine/utils.py`, `vllm/v1/engine/core.py`, `vllm/v1/engine/core_client.py` | Separate request drain from a named 15-second resource-cleanup bound; stop callback/I/O threads; detach roots; collect while the runtime is alive; always release MP frontend resources. Ray keeps its prior contract. |

`git grep 'torch.cuda.CUDAGraph(' HEAD -- 'vllm/**/*.py'` finds six direct graph
construction sites at the base. The current tree has one common construction
path. That supports the ownership/inventory design, but is not performance
evidence.

## Quantitative evidence

### Standalone rocprof trace

Local artifact set `rocprof-final-quiesced-20260819` contains a HIP API CSV that
recorded the following for 64 graphs in `explicit-reset-one-sync` mode:

- explicit reset phase: 64 `hipGraphExecDestroy` calls, 10.510629 ms total;
- explicit synchronization: one `hipDeviceSynchronize`, 0.008950 ms; and
- later Python wrapper destruction: 64 additional `hipDeviceSynchronize` calls,
  0.039413 ms total.

This verifies the current PyTorch destructor call pattern and the benchmark's
phase accounting. In this idle trace the additional synchronizations total only
0.039413 ms, so it does not establish them as the service shutdown bottleneck.
Phase attribution for the existing trace also relied on monotonic markers
recorded during the session; reruns should persist marker stderr next to the CSV.

The proposal's historical 128-graph `41.6/42.6 ms` comparison is intentionally
not repeated here: no raw artifact and exact command were found, so it is not
verified evidence.

An N=8 in-flight `rocgdb` smoke run is stored in the uncommitted local artifact
set `rocgdb-n8` (with another final session under
`rocgdb-final-current-20260819`); the inferior exited normally. This is a safety
smoke test, not a soak or performance result.

### Final TP=2 service trials

The two uncommitted local artifact run IDs are `e4f829f5` and `92c97ec8`.
Both used MI300X devices 1 and 2, `mp`, TP=2, FULL capture sizes
`[1, 2, 4, 8]`, child `--shutdown-timeout 0`, and the unset ROCm worker-grace
default (effective eight seconds). Each completed request returned HTTP 200 and
644 bytes.

| Measurement (ms) | `e4f829f5` | `92c97ec8` |
| --- | ---: | ---: |
| signal to API process reaped | 10,939.299691 | 10,150.721564 |
| last worker shutdown marker to reap | 9,765.510270 | 9,007.757696 |
| signal to EngineCore root detached | 7,415.978272 | 7,041.161850 |
| final `gc.collect()` marker span | 876.599374 | 452.979200 |
| signal to first exact 286 MiB/GPU sample | 7,041.976217 | 7,239.809525 |
| signal to two-second VRAM-stability completion | 13,669.226617 | 12,464.748463 |
| reap to VRAM-stability completion | 2,729.926926 | 2,314.026899 |

Each run observed eight `ModelCudaGraphManager` graphs, four per worker. No other
owner type was exercised by this end-to-end configuration.

| Run / rank | enumerate (ms) | reset four graphs (ms) | clear (ms) |
| --- | ---: | ---: | ---: |
| `e4f829f5` / 0 | 0.163440 | 1.728946 | 0.111491 |
| `e4f829f5` / 1 | 0.253779 | 1.995571 | 0.153015 |
| `92c97ec8` / 0 | 0.200115 | 1.826917 | 0.147162 |
| `92c97ec8` / 1 | 0.211550 | 1.941084 | 0.133641 |

Both result files report return code 0, `errors: []`, `forced_signal: null`, and
`process_tree_gone: true`; graph reset markers completed with zero failures and
owner clear retained zero owners. Worker logs say all workers exited gracefully.
Final samples returned to the exact 286 MiB-per-GPU baseline.

`forced_signal: null` means the outer harness did not escalate to SIGKILL. The
normal graceful path itself sends SIGTERM to the API process, so do not describe
these as "no SIGTERM" runs. The second run reused the same GPUs and port, but
started about 71 seconds after the first run's stability completion; call it a
subsequent restart, not an immediate restart.

### Controls that motivated the lifecycle bounds

- Local TP=2 run `4e82a89d`, with an explicit/default-equivalent five-second
  worker grace, logged that the parent sent SIGTERM to both workers after five
  seconds and reported worker exit codes `-15`.
- A 12-second TP=2 diagnostic (`5423210c`) completed gracefully.
- The two effective-eight-second runs above completed gracefully. These three
  observations support the bounded eight-second ROCm default when the setting is
  unset. They do not show that graph reset needs eight seconds; measured graph
  reset was about two milliseconds per worker.
- Earlier TP=1 full-stack diagnostic run IDs `7bf617b7` and `aa572ed8` contain
  `child.stdout.log` messages that the process manager was force-killing
  EngineCore. Their outer harness still recorded return code 0 and
  `forced_signal: null`, so the child log—not the outer result field—is the
  evidence of interrupted cleanup. These were intermediate stacked-candidate
  runs, not pristine-HEAD baselines.

The 15-second EngineCore resource-cleanup bound is defensive and intentionally
separate from the request-drain setting. It leaves margin over the 10.939-second
maximum of only two candidate observations; it is neither a latency target nor
a percentile-derived timeout.

## Proposed PR stack

| # | Draft | Proposed branch | Scope |
| ---: | --- | --- | --- |
| 1 | `pr_01_cuda_graph_teardown_tooling.md` | `akaratza_cuda_teardown_tooling` | Benchmarks and marker plumbing |
| 2 | `pr_02_scoped_cuda_graph_ownership.md` | `akaratza_cuda_teardown_ownership` | Ownership/reset primitives |
| 3 | `pr_03_register_cuda_graph_owners.md` | `akaratza_cuda_teardown_owners` | Convert non-ubatch graph owners |
| 4 | `pr_04_quiesce_ubatch_graphs.md` | `akaratza_cuda_teardown_ubatch` | Quiesce and reset ubatch graphs |
| 5 | `pr_05_reset_runner_graphs_before_model_state.md` | `akaratza_cuda_teardown_runners` | Runner and worker teardown order |
| 6 | `pr_06_multiproc_cuda_graph_shutdown.md` | `akaratza_cuda_teardown_multiproc` | Worker-process dependency order and grace |
| 7 | `pr_07_enginecore_root_detach.md` | `akaratza_cuda_teardown_enginecore` | EngineCore/frontend root lifetime and collection |

The final artifacts were captured from one dirty full-stack tree
(`changed_path_count: 40` in their metadata). They can support end-to-end safety
and explain design choices, but they do not isolate any one PR. Build each branch
from the base with explicit file/hunk selection, then rerun its focused tests and
the end-to-end scenario at the relevant stack point.

## Validation already run

The latest recorded broad reruns below passed at approximately 18:31 UTC on
2026-08-19. `gpu_worker.py` changed afterward and was separately covered by the
12-test rerun below plus static checks. Earlier iterative failures occurred
during development and were fixed; this is not a claim that every historical run
passed, that the broad counts reflect the literal final byte of every file, or
that the full repository test suite ran.

```bash
HIP_VISIBLE_DEVICES=3 pytest -q \
  tests/v1/cudagraph/test_cudagraph_teardown.py \
  tests/v1/cudagraph/test_breakable_cudagraph.py \
  tests/v1/cudagraph/test_encoder_cudagraph.py \
  tests/v1/worker/test_gpu_worker.py \
  tests/v1/worker/test_gpu_autoregressive_speculator.py
# 132 passed, 18 warnings in 67.53s
```

The warnings were 14 PyTorch JIT deprecations and four expected empty-HIP-graph
warnings.

```bash
HIP_VISIBLE_DEVICES=3 pytest -q \
  tests/v1/worker/test_gpu_model_runner.py \
  tests/v1/worker/test_gpu_model_runner_v2.py
# 67 passed, 2 skipped, 14 warnings in 62.60s

.venv/bin/python -m pytest -q \
  tests/v1/worker/test_gpu_worker.py \
  tests/utils_/test_shutdown_markers.py
# 12 passed, 14 warnings in 3.45s
```

Additional focused recorded results:

- `tests/v1/engine/test_startup_watch_processes.py`: 27 passed;
- `tests/v1/executor/test_worker_proc_shutdown.py`: 5 passed;
- focused executor termination/WorkerProc selection: 3 passed, 17 deselected;
- an earlier focused executor plus startup selection: 36 passed, 11 deselected;
- `benchmarks/shutdown_timing.py --self-test`: passed; and
- `ruff format --check` and `ruff check` across the 18 production graph/marker/
  speculator/worker files: passed; isolated `py_compile` also passed without
  leaving workspace bytecode; and
- later focused static checks covering the edited GPU worker and the five
  lifecycle files, plus `git diff --check`: passed on their then-current trees.

Because this is a shared dirty worktree, rerun static checks and the tests owned
by each stack layer immediately before committing it. Do not present the
aggregate counts as a current full-suite result.

## Environment and tools

- 8 x AMD Instinct MI300X OAM (`gfx942`, approximately 192 GiB each)
- ROCm 7.2.3; HIP 7.2.53211
- amdgpu reported by AMD SMI: 6.16.6; kernel: 5.15.0-186-generic
- PyTorch 2.12.0+git6bbd260
- CPython 3.12.13; pytest 9.1.1; Ruff 0.16.1
- rocprofv3 1.1.0 (`c2d9476115...`)
- rocgdb / GNU gdb 16.3 (`rocm-rel-7.2-90`)
- AMD SMI 26.2.2+c2d9476115

The required tools were already installed; no installation was needed.

## Reproduction

Run these from the repository root. The commands deliberately write new output
under `/tmp`; do not rely on the local evidence directories being available to
reviewers.

Standalone graph trace:

```bash
mkdir -p /tmp/rocprof-cudagraph-n64
HIP_VISIBLE_DEVICES=3 rocprofv3 --runtime-trace --stats --summary \
  --summary-output-file /tmp/rocprof-cudagraph-n64/summary.txt \
  -d /tmp/rocprof-cudagraph-n64 -f csv -- \
  .venv/bin/python benchmarks/cudagraph_lifetime.py \
  -n 64 --mode explicit-reset-one-sync --device cuda:0 \
  >/tmp/rocprof-cudagraph-n64/result.json \
  2>/tmp/rocprof-cudagraph-n64/phase-markers.jsonl
```

Final TP=2 lifecycle scenario (run twice to exercise a subsequent same-GPU/port
restart):

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

Here the harness's 120-second `--shutdown-timeout` is an outer safety bound; the
child's `--shutdown-timeout 0` disables request draining. The internal 15-second
EngineCore resource-cleanup grace is a separate implementation constant.

Harness self-test:

```bash
.venv/bin/python benchmarks/shutdown_timing.py --self-test
```

## Important caveats and deferred work

- `vllm/compilation/cuda_graph.py` grew from 361 to 1,378 lines
  (+1,036/-19), and the new dedicated teardown test is 1,500 lines. The module
  now contains substantially more exception policy and ownership machinery than
  before. Before posting PR 2, strongly consider extracting the ownership/reset
  framework into a focused module and reducing broad exception boundaries. The
  current measurements do not justify landing roughly 1,000 framework lines in
  the existing graph module as one change.
- The user explicitly prefers Pythonic, straight-line shutdown code with few
  catches. Keep catches only at terminal cleanup/resource boundaries where
  failure precedence or best-effort cleanup requires them; log or propagate each
  failure. Re-audit every remaining catch per stacked PR.
- `gc.freeze()` remains an exit-only ROCm fallback. This work does not remove it.
- The API output-handler can still log a pre-existing `EngineDeadError` after an
  otherwise clean shutdown; that is deferred.
- No PyTorch patch is included. A conditional upstream destructor change remains
  deferred until vLLM ownership is proven and a real bottleneck is measured. The
  proposed old/new ROCm safety boundary (including 6.3.1/6.3.2) was not validated
  locally.
- Not completed: MI355 coverage, NVIDIA regression coverage, 30/50-trial
  distributions, 100-cycle soak, no-freeze comparison, same-process recapture,
  active-request shutdown, Ray teardown, PIECEWISE owner coverage, or all owner
  types in the service-level harness.
- The benchmark artifact metadata identifies a dirty full-stack tree rather than
  an immutable commit. Capture clean-branch evidence for final PR descriptions.
- Local artifact directories and machine-specific identifiers must not be
  committed. Sanitize logs and upload only deliberate evidence bundles.

## Suggested next steps

1. Review this handoff and all seven PR drafts against the live diff. Obtain the
   external proposal from the original task owner only if needed, and treat
   measured artifacts—not proposal language—as authoritative.
2. Reduce/extract the ownership framework and re-audit exception handling before
   freezing PR 2.
3. Create the seven stacked branches from the recorded base with explicit
   file/hunk staging. Keep unrelated shared-worktree changes out.
4. Rerun each layer's focused unit/static checks on its isolated branch; rerun
   the TP=2 lifecycle scenario where worker and EngineCore behavior enters.
5. Add controlled baseline/candidate trials and repeated/soak coverage before
   making any performance claim.
6. Sanitize and publish raw evidence, replace local artifact references in the
   PR drafts with reviewer-accessible links, and review the resulting diffs.
7. This WIP snapshot was committed and pushed only after explicit user approval.
   Do not open a PR or add follow-up commits without another deliberate review.
