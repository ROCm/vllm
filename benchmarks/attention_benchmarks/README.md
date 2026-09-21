# vLLM Attention Benchmarking Suite

Fast, flexible benchmarking for vLLM attention and MLA backends with an extended batch specification grammar.

## Quick Start

```bash
cd benchmarks/attention_benchmarks

# Run a pre-configured benchmark
python benchmark.py --config configs/mla_decode.yaml
python benchmark.py --config configs/mla_mixed_batch.yaml
python benchmark.py --config configs/speculative_decode.yaml
python benchmark.py --config configs/standard_attention.yaml
python benchmark.py --config configs/reorder_threshold.yaml

# Or run custom benchmarks
python benchmark.py \
    --backends flash flashinfer \
    --batch-specs "q2k" "8q1s1k" "2q2k_32q1s1k" \
    --output-csv results.csv
```

## Cache residency

`benchmark_fn` walks one KV cache per layer and divides the result by the layer
count, so the layer loop doubles as a rotation over disjoint KV regions. Whether
a measurement reads DRAM or a resident last-level cache therefore depends on
`num_layers * kv_bytes_per_layer`, not on one layer's cache.

The default of 10 layers is enough for most shapes. It stops being enough when
the product falls below the cache, which on a Strix Halo means a 32 MiB MALL
that CUDA-graph replay never flushes. Measured on that board, TRITON_ATTN
decode, with and without the option:

| working set (10 layers) | cells | median bias | worst |
| --- | --- | --- | --- |
| under 32 MiB | 6 | +35% | +64% |
| over 32 MiB | 18 | +0% | +3% |

The split is clean: every cell that fits the MALL was reporting a time too low
to be sustainable, and every cell that does not was already fine. Small models
at short contexts are the ones affected — `32x8x128` at 512 tokens reads 20 MB
across ten layers and came out 64% fast.

`--min-working-set-mb` raises `--num-layers` until the KV working set reaches
the given size. It only ever raises, so shapes already past the cache are
untouched and their timings do not move.

```bash
# 20 MB across the default 10 layers: inside the MALL.
python benchmark.py --backends TRITON_ATTN --batch-specs q1s512 \
    --num-q-heads 32 --num-kv-heads 8 --head-dim 128      # 11 us, cached
python benchmark.py --backends TRITON_ATTN --batch-specs q1s512 \
    --num-q-heads 32 --num-kv-heads 8 --head-dim 128 \
    --min-working-set-mb 96                               # 18 us, DRAM
```

## Simplified Batch Specification Grammar

Express workloads concisely using query length and sequence length:

```python
"q2k"              # 2048-token prefill (q_len=2048, seq_len=2048)
"q1s1k"            # Decode: 1 token with 1K sequence
"8q1s1k"           # 8 decode requests
"q4s1k"            # 4-token extend (e.g., spec decode)
"2q2k_32q1s1k"     # Mixed: 2 prefills + 32 decodes
"16q4s1k"          # 16 spec decode (4 tokens each)
```

### Grammar Rule

```text
Format: (<count>?) q<q_len>(k?) (s<seq_len>(k?))?

- count:   Number of identical requests (optional, default=1)
- q_len:   Query length (number of new tokens)
- seq_len: Total sequence length (optional, defaults to q_len for prefill)
- 'k':     Multiplies value by 1024

Mixed batches: Use _ to combine (e.g., "2q2k_32q1s1k")
```

**Note**: Decode, prefill, and spec decode are just different query lengths - no special syntax needed!

## Pre-configured Benchmarks

The suite includes several pre-configured YAML benchmark configurations:

### MLA Decode Benchmark

Tests pure decode performance across MLA backends with varying batch sizes and sequence lengths.

```bash
python benchmark.py --config configs/mla_decode.yaml
```

### MLA Mixed Batch Benchmark

Tests chunked prefill performance with mixed prefill + decode batches.

```bash
python benchmark.py --config configs/mla_mixed_batch.yaml
```

### Speculative Decoding Benchmark

Tests speculative decode scenarios (K-token verification) and reorder_batch_threshold optimization.

```bash
python benchmark.py --config configs/speculative_decode.yaml
```

### Standard Attention Benchmark

Tests standard attention backends (Flash/Triton/FlashInfer) with pure prefill, decode, and mixed batches.

```bash
python benchmark.py --config configs/standard_attention.yaml
```

### Reorder Threshold Study

**Question:** At what query length does the prefill pipeline become faster than the decode pipeline?

Tests query lengths from 1-1024 across 9 batch sizes to find the crossover point. Uses `decode_vs_prefill` mode to compare both pipelines for each query length.

```bash
python benchmark.py --config configs/reorder_threshold.yaml
```

---

## Universal Benchmark

The `benchmark.py` script handles **all** backends - both standard attention and MLA.

### Standard Attention (Flash/Triton/FlashInfer)

```bash
python benchmark.py \
    --backends flash triton flashinfer \
    --batch-specs "q2k" "8q1s1k" "2q2k_32q1s1k" \
    --num-layers 10 \
    --output-csv results.csv
```

### MLA Backends

```bash
# Compare all MLA backends
python benchmark.py \
    --backends cutlass_mla flashinfer_mla flashattn_mla flashmla \
    --batch-specs "64q1s1k" "64q1s4k" \
    --output-csv mla_results.csv
```

### Parameter Sweeps

Use `--sweep-param` and `--sweep-values` to run parameter sweeps from the CLI:

#### CUTLASS MLA num-splits Optimization

**Question:** What is the optimal `num_kv_splits` for CUTLASS MLA?

```bash
python benchmark.py \
    --backend cutlass_mla \
    --batch-specs "64q1s1k" "64q1s4k" "64q1s16k" \
    --sweep-param num_kv_splits \
    --sweep-values 1 2 4 8 16 \
    --output-json optimal_splits.json
```

#### Reorder Batch Threshold Optimization

**Question:** What's the optimal `reorder_batch_threshold` for speculative decoding?

```bash
python benchmark.py \
    --backend flashmla \
    --batch-specs "q4s1k" "q8s2k" \
    --sweep-param reorder_batch_threshold \
    --sweep-values 1 4 16 64 256 512 \
    --output-csv threshold_sweep.csv
```

### All Command-Line Options

```text
--config CONFIG                     # Path to YAML config file (overrides other args)
--backends BACKEND [BACKEND ...]    # flash, triton, flashinfer, cutlass_mla,
                                    # flashinfer_mla, flashattn_mla, flashmla
--backend BACKEND                   # Single backend (alternative to --backends)
--batch-specs SPEC [SPEC ...]       # Batch specifications using extended grammar

# Model configuration
--num-layers N                      # Number of layers
--head-dim N                        # Head dimension
--v-head-dim N                      # Value head dimension (defaults to --head-dim)
--num-q-heads N                     # Query heads
--num-kv-heads N                    # KV heads
--block-size N                      # Block size
--kv-lora-rank N                    # MLA KV LoRA rank
--qk-nope-head-dim N                # MLA non-RoPE QK head dim
--qk-rope-head-dim N                # MLA RoPE QK head dim

# Benchmark settings
--device DEVICE                     # Device (default: cuda:0)
--warmup-ms N                       # Warmup window in ms for triton do_bench
--profile-memory                    # Profile memory usage

# Parameter sweeps
--sweep-param PARAM                 # Parameter name to sweep (e.g., num_kv_splits,
                                    # reorder_batch_threshold)
--sweep-values N [N ...]            # Values to sweep for the parameter

# Output
--output-csv FILE                   # Save to CSV
--output-json FILE                  # Save to JSON
```

## Hardware Requirements

| Backend | Hardware |
| ------- | -------- |
| Flash/Triton/FlashInfer | Any CUDA GPU |
| CUTLASS MLA | Blackwell (SM100+) |
| FlashAttn MLA | Hopper (SM90+) |
| FlashMLA | Hopper (SM90+) |
| FlashInfer-MLA | Any CUDA GPU |

## Using MLA Runner Directly

All MLA backends are available through `mla_runner.run_mla_benchmark()`:

```python
from mla_runner import run_mla_benchmark
from common import BenchmarkConfig

config = BenchmarkConfig(
    backend="cutlass_mla",
    batch_spec="64q1s4k",
    num_layers=10,
    head_dim=576,
    num_q_heads=128,
    num_kv_heads=1,
    block_size=128,
    device="cuda:0",
)

# CUTLASS MLA with specific num_kv_splits
result = run_mla_benchmark("cutlass_mla", config, num_kv_splits=4)
print(f"Time: {result.mean_time:.6f}s")

# FlashInfer-MLA
result = run_mla_benchmark("flashinfer_mla", config)

# FlashAttn MLA (Hopper SM90+)
result = run_mla_benchmark("flashattn_mla", config, reorder_batch_threshold=64)

# FlashMLA (Hopper SM90+)
result = run_mla_benchmark("flashmla", config, reorder_batch_threshold=64)
```

## Python API

```python
from batch_spec import parse_batch_spec, format_batch_spec, get_batch_stats
from common import BenchmarkConfig, BenchmarkResult, ResultsFormatter

# Parse batch specs
requests = parse_batch_spec("2q2k_q4s1k_32q1s1k")
print(format_batch_spec(requests))
# "2 prefill (2x2k), 1 extend (1xq4kv1k), 32 decode (32x1k)"

# Get batch statistics
stats = get_batch_stats(requests)
print(f"Total tokens: {stats['total_tokens']}")
print(f"Num decode: {stats['num_decode']}, Num prefill: {stats['num_prefill']}")

# Format results
formatter = ResultsFormatter()
formatter.save_csv(results, "output.csv")
formatter.save_json(results, "output.json")
```

## Tips

**1. Save results** - Always use `--output-csv` or `--output-json`

**2. Test incrementally** - Start with `--num-layers 1`

**3. Extended grammar** - Leverage spec decode, chunked prefill patterns

**4. Parameter sweeps** - Use `--sweep-param` and `--sweep-values` to find optimal values
