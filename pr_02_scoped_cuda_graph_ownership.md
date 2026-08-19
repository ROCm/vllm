Title: [Core] Add scoped CUDA graph ownership and reset
Proposed branch: `akaratza_cuda_teardown_ownership`
Depends on: `akaratza_cuda_teardown_tooling`

# Description

`CUDAGraphWrapper` and breakable graph cleanup currently clear Python dictionaries and leave native destruction to reference counting or cyclic GC, so shutdown cannot control ordering or attribute the cost. This change introduces an engine-scoped owner contract and a two-phase coordinator: close owners to new capture/replay, materialize and deduplicate their graphs by identity and device, synchronize once per device, reset every safe graph, and only then clear owner state. Two clean MI300 TP2 runs enumerated four graphs per worker; across both ranks, enumeration took 0.16-0.25 ms, reset took 1.73-2.00 ms, and owner clear took 0.11-0.15 ms. Separately, an `N=64` trace measured the 64 post-reset wrapper-destructor synchronizations at only 0.039 ms total. These measurements motivate deterministic ownership and ordering, not an exit-latency speedup.

- Add `OwnedCUDAGraph`, the owner protocol, teardown statistics, and an idempotent scoped registry.
- Route `CUDAGraphWrapper` and breakable segment construction through one tracked factory and retain explicit breakable graph identity.
- Support partial capture, reusable reset, multi-device grouping, missing-reset capability errors, and bounded terminal retention without relying on `__del__` ordering.
- Add fake-graph tests for ordering, deduplication, failure handling, concurrency, engine isolation, and repeated teardown.

The existing weak-set cleanup remains as a temporary compatibility bridge in this stack and is removed only when the runner activation PR switches every production caller to scoped ownership.

Local full-stack evidence uses run IDs `e4f829f5` and `92c97ec8`. These runs include the later runner, multiprocessing, and EngineCore PRs, so they validate the protocol's end-to-end accounting rather than isolating this PR's performance. The artifacts and phase markers are not committed and must be attached somewhere reviewers can access before posting.
