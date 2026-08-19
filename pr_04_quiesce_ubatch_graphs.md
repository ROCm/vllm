Title: [Core] Make ubatch CUDA graph capture quiesceable
Proposed branch: `akaratza_cuda_teardown_ubatch`
Depends on: `akaratza_cuda_teardown_owners`

# Description

UBatch owns both direct graphs and a nested `CUDAGraphWrapper`, while its capture and replay work runs across barrier-coordinated Python threads. Treating it as an ordinary dictionary owner can reset a graph while a sibling thread is still entering capture or replay. This change gives UBatch one localized thread-lifecycle boundary and registers the outer owner before its child, allowing teardown to reject new work and wait for already-admitted work without adding exception scaffolding to `UBatchContext`.

- Register `UBatchContext` globally only after `__enter__` initialization succeeds.
- Centralize thread start, barrier-abort, wake, join, and outstanding-thread tracking in `UBatchWrapper`.
- Enumerate direct and nested graphs while retaining stuck-thread state instead of resetting underneath it.
- Cover failed thread start, broken barriers, context-entry failure, stuck threads, parent/child quiescence order, and real MI300 capture/replay/teardown.

Normal non-ROCm forward joins remain unbounded as before; only ROCm failure cleanup and terminal quiescence use the bounded wait.
