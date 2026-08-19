Title: [Core] Register every non-ubatch CUDA graph owner
Proposed branch: `akaratza_cuda_teardown_owners`
Depends on: `akaratza_cuda_teardown_ownership`

# Description

The common graph wrapper is only part of vLLM's ownership surface: full model graphs, encoder budget graphs, Gemma4 centroid graphs, and speculative managers also retain native graphs and static buffers. At the base revision, `git grep 'torch.cuda.CUDAGraph(' HEAD -- 'vllm/**/*.py'` finds six direct construction sites; after this conversion, the working tree has one common factory. This change puts those paths under the same owner contract so each owner releases replay callables and buffers only after native reset.

- Add deterministic enumeration and idempotent clear hooks to full/breakable model managers, encoder managers, and Gemma4.
- Store graph metadata before capture setup so failed or partial initialization remains visible to teardown.
- Add base and concrete speculative-manager shutdown delegation without changing proposal or replay behavior.
- Enforce centralized native graph construction and test every converted owner with fake graphs plus real MI300 encoder capture/replay/clear coverage.

This PR does not activate runner shutdown ordering; it makes the owner inventory complete and independently testable before the runtime switch.
