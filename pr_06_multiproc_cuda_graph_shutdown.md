Title: [ROCm] Make multiprocessing graph shutdown dependency-safe
Proposed branch: `akaratza_cuda_teardown_multiproc`
Depends on: `akaratza_cuda_teardown_runners`

# Description

Multiprocessing workers perform graph cleanup from `WorkerProc`'s terminal `finally`, but the async output thread, message queues, distributed groups, and parent exit checks previously had no shared dependency order. A worker could therefore reset graphs while output work was still active, destroy distributed state after an incomplete reset, or exit nonzero while the parent reported a clean shutdown. This change makes the worker-process boundary explicit without broadening normal executor behavior.

- Stop and join async output handling before worker graph teardown, then close message queues and distributed state in dependency order.
- Run worker shutdown under the terminal graph policy and retain incomplete graph state instead of destroying its process groups.
- Check worker exit codes after independent queue cleanup and surface forced or nonzero termination to EngineCore.
- Add three seconds to the existing five-second worker grace only for ROCm's unset default. In an explicit TP=2 MI300X five-second control, both workers were still finalizing at the deadline and the parent escalated them with SIGTERM (`VllmWorker-0=-15`, `VllmWorker-1=-15`); two runs with the effective eight-second default instead logged that all workers exited gracefully. Explicit settings and non-ROCm behavior remain unchanged.
- Add focused WorkerProc and executor failure-order tests, including queue failures and async-output quiescence.

EngineCore signal, callback, and root lifetime remain unchanged here and are handled in the final stack PR.

Local stacked-candidate evidence: the five-second control is run `4e82a89d`; the two eight-second runs are `e4f829f5` and `92c97ec8`. Each final run enumerated four graphs per worker and the reset-marker spans were only 1.73-2.00 ms, so the added grace is for worker-process finalization, not graph-reset latency. These uncommitted end-to-end artifacts include the later EngineCore stack PR and therefore support the timeout choice rather than isolating this PR's performance; attach them somewhere reviewers can access before posting.
