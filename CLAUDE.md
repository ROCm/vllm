# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

---

## Architecture Overview

vLLM is a high-throughput LLM inference engine built around **PagedAttention** (non-contiguous KV cache memory), **continuous batching**, and a multi-process architecture. The active production path lives in `vllm/v1/`.

### Multi-Process Design

A running vLLM server spawns several process types that communicate over ZMQ:

| Process | Count | Role |
|---|---|---|
| API Server | 1 per DP rank | HTTP, tokenization, multi-modal feature extraction |
| Engine Core | 1 per DP rank | Scheduler, KV cache manager, busy-loop dispatch |
| GPU Worker | 1 per GPU (TP × PP × DP) | Model weights, forward pass, sampling |
| DP Coordinator | 1 (if DP > 1) | Cross-rank load balancing for MoE models |

Example: 4 GPUs with TP=4 → 1 API server + 1 engine core + 4 workers = 6 processes.

### Request Lifecycle

```
HTTP Request
  → API Server (tokenize, multimodal encoding)
  → EngineCore (ZMQ) → Scheduler → KV cache allocation
  → GPU Workers (forward pass → sampling)
  → EngineCore (logprobs, detokenization)
  → API Server (streaming response)
```

### Key Subsystems

**Scheduler** (`v1/core/sched/scheduler.py`) — Continuous batching engine. Separates requests into prefill (new prompts) and decode (ongoing generation) phases. Chunked prefill splits large prompts across iterations. Produces a `SchedulerOutput` each iteration containing block IDs, token counts, and new/continuing request metadata.

**KV Cache Manager** (`v1/core/kv_cache_manager.py`, `v1/core/kv_cache_utils.py`) — Implements paged memory: the KV cache is divided into fixed-size blocks that can be non-contiguous (like virtual memory pages). Prefix caching reuses blocks via content-based hashing (SHA256 of parent hash + token IDs). Identical prefixes across requests automatically share cached blocks.

**GPU Model Runner** (`v1/worker/gpu_model_runner.py`) — Orchestrates the forward pass: batches tokens from multiple requests into `GPUInputBatch`, calls the model, runs sampling. Captures CUDA graphs for the decode phase to reduce CPU-GPU sync overhead.

**Attention Backends** (`v1/attention/`, `attention/`) — Pluggable via `--attention-backend`. Key implementations: FlashAttention, FlashInfer, Paged Attention (vLLM's custom CUDA kernel), Triton. Auto-selected based on hardware. The `AttentionBackend` base class defines the interface all backends implement.

**Model Executor** (`model_executor/`) — Houses model architecture implementations (`models/`), optimized layer implementations (`layers/`), and weight loading (`model_loader/`). Supports 200+ model architectures loaded directly from HuggingFace checkpoints. The layer system is pluggable — custom layers can be registered and swapped for quantized or platform-specific variants.

**Distributed Execution** (`distributed/`, `config/parallel.py`) — Supports tensor parallel (TP), pipeline parallel (PP), data parallel (DP), expert parallel (EP for MoE), and context parallel (CP). Uses NCCL for GPU collectives. `parallel_state.py` manages `torch.distributed` process groups. MoE models use all-to-all shuffling to route tokens to expert shards.

**Configuration** (`config/`) — `VllmConfig` is the top-level container. Key sub-configs: `ModelConfig`, `ParallelConfig`, `SchedulerConfig`, `CacheConfig`, `AttentionConfig`, `CompilationConfig`, `SpeculativeConfig`. All are passed through the system rather than accessed via globals.

### Entry Points

- **`entrypoints/llm.py`** — Offline `LLM` class for single-process Python usage.
- **`entrypoints/openai/`** — OpenAI-compatible HTTP API server (FastAPI). Handles `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, streaming.

### Legacy vs. V1 Engine

`vllm/engine/` contains the legacy async/sync engine interfaces; they now delegate to the V1 engine. New work should target `vllm/v1/`. The V1 path is enabled by default.

### Domain-Specific Guides

Before editing these areas, read the linked guide:

- **Attention backends**: `docs/design/attention_backends.md`
- **Paged attention / KV cache**: `docs/design/paged_attention.md`
- **Prefix caching**: `docs/design/prefix_caching.md`
- **Architecture overview**: `docs/design/arch_overview.md`
- **Multimodal inputs**: `docs/design/multimodal/`
