Title: [ROCm] Reset runner graphs before releasing model state
Proposed branch: `akaratza_cuda_teardown_runners`
Depends on: `akaratza_cuda_teardown_ubatch`

# Description

Both model runners currently release graph managers, KV caches, models, and workspaces through separate implicit-reference paths. This change gives each runner a private registry before model construction and makes ROCm shutdown reset that engine's graphs before speculative managers, static buffers, KV state, model state, allocator pools, or workspaces are released. Temporary profiling graphs reuse the same coordinator without closing serving owners, and an incomplete reset retains its dependent state for terminal fallback instead of reporting the engine reusable.

In two final-stack TP=2 MI300X runs, each runner enumerated all four captured `ModelCudaGraphManager` graphs with zero teardown failures; the four per-worker reset-marker spans were 1.73-2.00 ms. That measurement validates the intended ownership and release order. It is not evidence that explicit reset shortens the overall 10.15-10.94 second process shutdown.

- Create isolated V1 and V2 runner registries without routing late compiled owners into another engine.
- Apply the same reset/clear ordering to full, piecewise, encoder, ubatch, and speculative graphs.
- Reuse scoped reset for profiling graphs, restore graph pools after successful cleanup, and reject reuse after incomplete teardown.
- Propagate the incomplete outcome through GPUWorker and UniProc while preserving the established CUDA/XPU/CPU fail-fast path.
- Replace the legacy process-wide weak-set sweeps and cover ordering, isolation, profiling, fallback, and non-ROCm behavior in focused runner tests.

The existing exit-only freeze remains a fallback; this PR does not remove it or change Ray actor teardown.

Focused validation on MI300X: the complete legacy and V2 runner test files passed 67 tests with 2 skips. The end-to-end numbers come from the two result artifacts cited in the final EngineCore PR draft and therefore include later stack layers.
