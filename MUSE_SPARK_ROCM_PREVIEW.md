# Muse Spark Open on ROCm (MI355X): optimization preview

This branch is a preview of the Muse Spark Open optimizations for ROCm (MI355X, gfx950) that are under review in vLLM. The open PRs are merged on `main` at [`bf5567a306`](https://github.com/vllm-project/vllm/commit/bf5567a306) (2026-11-17).

It is not for production. The branch is rebuilt as the PRs change. Please send feedback on the individual PRs.

## Contents

| PR | title | head | knob |
| --- | --- | --- | --- |
| [#67133](https://github.com/vllm-project/vllm/pull/67133) | AITER FA: split-KV kernels for decode and speculative-decode verify | `9e20f1b71b` | always on (gate below) |
| [#67145](https://github.com/vllm-project/vllm/pull/67145) | AITER FA: FP8-query prefill attention | `83a4decec3` | `--attention-config '{"use_prefill_query_quantization": true}'` |
| [#67381](https://github.com/vllm-project/vllm/pull/67381) | Online `fp8_per_channel` dense linears on tuned FP8 kernels | `0be24e50c1` | `--quantization-config` (below) |
| [#67191](https://github.com/vllm-project/vllm/pull/67191) | Online MXFP4 for the BF16 first/last-layer experts | `d578da7732` | `--quantization-config` (below) |
| [#67306](https://github.com/vllm-project/vllm/pull/67306) | Decode MoE on native routes | `caae34db45` | always on (gate below) |
| [#67380](https://github.com/vllm-project/vllm/pull/67380) | Ulysses sequence parallelism for TEP prefill | `4fb0a7b070` | `--additional-config '{"ulysses_prefill": true}'` |
| [#67153](https://github.com/vllm-project/vllm/pull/67153) | Seed the fast detokenizer with a prompt tail | `c2f685aa8c` | always on |
| [#67382](https://github.com/vllm-project/vllm/pull/67382) | Fuse the DSpark Markov sampling steps | `f3c1f23a12` | always on (gate below) |
| [#67340](https://github.com/vllm-project/vllm/pull/67340) | FP8 DSpark draft through the speculative config | `1001c6467c` | `--speculative-config` keys (below) |

PRs #67306, #67380 and #67381 are a stack: #67381 contains #67380, which contains #67306.

Already in `main`, so no longer listed as dependencies:

- [#63085](https://github.com/vllm-project/vllm/pull/63085): token-sharded residual stream across TP (always on; env `VLLM_ROCM_MUSE_SP_MIN_TOKENS`, default 1024, `0` = off).
- [#66051](https://github.com/vllm-project/vllm/pull/66051): DSpark Markov sampling as one launch per round under `--monokernel`.
- [#65985](https://github.com/vllm-project/vllm/pull/65985): DSpark draft layers as one launch under `--monokernel`. It needs BF16 draft weights; with #67340's FP8 draft it logs `drafter off` and the stock draft layers run.
- [#59098](https://github.com/vllm-project/vllm/pull/59098): AITER FA context fix for out-of-order batches.

## Knobs, by measured impact

| PR | default | knob | gate | measured impact (MI355X) |
| --- | --- | --- | --- | --- |
| #67133 | on | none | gfx950, `ROCM_AITER_FA`, FP8 KV, head 64, GQA 8, decode q_len 1-8, FlyDSL | AgentX TEP4 vs main: c24 +54% tok/s/GPU (ITL 21.0 -> 7.4 ms), c40 +99% (41.5 -> 11.3 ms) |
| #67145 | off | `--attention-config '{"use_prefill_query_quantization": true}'` | gfx950, FP8 e4m3 KV, full-attention layers | AgentX c24 +3.1%, c40 +5.4%; prefill TTFT -13..-16% |
| #67381 | off | `--quantization-config '{"linear": "fp8_per_channel", "ignore": ["*o_proj"]}'` | gfx950, not with `--monokernel` | decode step -8% (1 request), -5% (8-24) |
| #67191 | off | `--quantization-config '{"targets": {"*layers.0.feed_forward.experts*": "mxfp4", "*layers.61.feed_forward.experts*": "mxfp4"}}'` | routed experts only | AgentX c24 +2.8%, c40 +3.0% (ITL -7%) |
| #67306 | on | none | gfx950, EP, MXFP4 experts, <= 2048 tokens per call | decode step -4..-5% |
| #67380 | off | `--additional-config '{"ulysses_prefill": true}'` | TEP (TP > 1 with EP), gfx950 | TTFT 8k/16k -18%; KV -12 GiB per GPU |
| #67153 | on | none | all models | AgentX TTFT -7..-12% (c10-c24) |
| #67382 | on | none | gfx950, not `--monokernel` | decode step -1..-2% |
| #67340 | off | `--speculative-config` keys `"quantization": "fp8_per_channel"`, `"draft_lm_head_quantization": "fp8"` | ROCm | decode step -1% (1 request) |

To use #67381 and #67191 together, put both in one `targets` map (a `targets` map replaces `linear`, so do not combine the two keys):
`{"targets": {"re:.*\\.(qkv_proj|pooler|unpooler)$": "fp8_per_channel", "*layers.0.feed_forward.experts*": "mxfp4", "*layers.61.feed_forward.experts*": "mxfp4"}}`.

## Recommended launch commands

All three use these environment variables and common arguments:

```bash
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_MUSE_MOE_REDUCE_DTYPE=bf16 VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_KV_CACHE_LAYOUT=LBHNC
M=amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4
COMMON=(--reasoning-parser muse --enable-auto-tool-choice --tool-call-parser muse --dtype bfloat16
  --language-model-only --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
  --enable-prefix-caching --enable-chunked-prefill --attention-backend ROCM_AITER_FA --max-model-len 1048576)
SPEC='"method":"dspark","model":"Inferact/Muse-Spark-Open-DSpark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":false,"attention_backend":"ROCM_AITER_FA"'
QCFG='{"targets":{"re:.*\\.(qkv_proj|pooler|unpooler)$":"fp8_per_channel","*layers.0.feed_forward.experts*":"mxfp4","*layers.61.feed_forward.experts*":"mxfp4"}}'
```

**TEP4, throughput (concurrency 1-40 per 4 GPUs).** Set `--max-num-seqs` to about 2x the concurrency.

```bash
vllm serve $M --tensor-parallel-size 4 --enable-expert-parallel --max-num-seqs 48 "${COMMON[@]}" \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.93 --max-num-batched-tokens 16384 --prefill-schedule-interval 48 \
  --speculative-config "{$SPEC,\"quantization\":\"fp8_per_channel\",\"draft_lm_head_quantization\":\"fp8\"}" \
  --additional-config '{"ulysses_prefill": true}' --attention-config '{"use_prefill_query_quantization": true}' \
  --quantization-config "$QCFG"
```

**TEP8, lowest latency (1-2 running requests).** Use `--max-num-seqs 1` (about 500 tok/s per user) or `2`. This mode uses the monokernel with a BF16 KV cache and 16-token blocks. #67133, #67145 and #67381 do not apply to it. The results below use the FP8 draft; to run #65985's one-launch draft instead, drop the draft `"quantization"` key (not measured on this branch).

```bash
vllm serve $M --tensor-parallel-size 8 --enable-expert-parallel --monokernel --block-size 16 --max-num-seqs 1 "${COMMON[@]}" \
  --gpu-memory-utilization 0.90 \
  --speculative-config "{$SPEC,\"quantization\":\"fp8_per_channel\"}" \
  --quantization-config '{"targets":{"*layers.0.feed_forward.experts*":"mxfp4","*layers.61.feed_forward.experts*":"mxfp4"}}'
```

**TEP8, FP8 stack, cap 8 (about 240 tok/s per user).**

```bash
vllm serve $M --tensor-parallel-size 8 --enable-expert-parallel --max-num-seqs 8 "${COMMON[@]}" \
  --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 --max-num-batched-tokens 16384 --prefill-schedule-interval 48 \
  --speculative-config "{$SPEC,\"quantization\":\"fp8_per_channel\",\"draft_lm_head_quantization\":\"fp8\"}" \
  --attention-config '{"use_prefill_query_quantization": true}' --quantization-config "$QCFG"
```

## Results

How these were measured:

- Measured 2026-11-13 on the previous build of this preview (merge head `cd433c4ed0`: earlier heads of the same PRs, #63085 and #66051 as PR heads, and the GateLinear reload fix) with the launch commands above. Not re-measured on this build.
- AgentX agentic-coding traces, one 1-hour run per point (9 runs).
    - TEP4 runs use 4 GPUs, two servers per 8x MI355X node.
    - TEP8 runs use a whole node.
- DSpark with 7 draft tokens and synthetic acceptance length 4.05, as in the InferenceX CI.
- Interactivity = median tok/s per user, as on the InferenceX dashboard.
- vLLM main = `083fbac02` with an FP8 KV cache and the same scheduler settings, without these PRs.
- Product = (throughput ratio) x (ITL ratio), at ITL p50, against SGLang's public InferenceX 1106 points at the same concurrency. Above 1 means faster.

| point | tok/s/GPU | ITL p50 / p90 (ms) | interactivity (tok/s/user) | vs vLLM main (tok/s/GPU) | product vs SGLang-1106 |
| --- | --- | --- | --- | --- | --- |
| TEP4 c10 | 16,522 | 4.00 / 4.79 | 246 | +25% | 1.10 (vs TP8 c10 cap 2) |
| TEP4 c16 | 26,114 | 4.52 / 6.00 | 217 | +63% | - |
| TEP4 c24 | 38,583 | 5.63 / 7.73 | 174 | +91% | - |
| TEP4 c28 | 42,189 | 6.14 / 8.97 | 159 | +100% | 1.03 |
| TEP4 c36 | 49,144 | 7.08 / 10.53 | 137 | - | 1.28 |
| TEP4 c40 | 51,088 | 7.68 / 11.69 | 127 | +152% | 1.41 |
| TEP8 c6, cap 1, monokernel | 4,968 | 2.00 / 2.10 | 497 | - | 0.84 |
| TEP8 c10, cap 2, monokernel | 8,375 | 2.85 / 3.21 | 347 | - | 0.78 |
| TEP8 c28, cap 8, FP8 stack | 22,868 | 4.09 / 5.40 | 238 | - | 0.99 |

## Accuracy

- Measured on earlier integrations (2026-11-11/13). GSM8K 0.977 with real acceptance (acceptance length 3.95).
- Needle-in-a-haystack 48/48.
- MATH-500 0.970 vs 0.969 on main (12 runs, p = 0.72; earlier integration without #67191).
- Each PR reports its own accuracy checks.

## Known limitations

- gfx950 (MI355X) only. Tested with TEP4 and TEP8 (`--enable-expert-parallel`); other layouts fall back or raise.
- Layerwise `reload_weights` with the gfx950 bf16x3 router GEMM raises `KeyError: attribute '_bf16x3_weight' already exists`; the fix is not posted yet and not in this branch. Reloading with DSpark speculative decoding and CUDA graphs can also produce garbled output (pre-existing on main; fine with `--enforce-eager` or without speculative decoding).
- The monokernel needs `--block-size 16` and a BF16 KV cache.
- Under `--monokernel` the FP8 dense linears (#67381) raise, and #67133/#67145 do not engage (BF16 KV).
- Requires recent AITER and FlyDSL; #67133 and the monokernel compile FlyDSL kernels on first use. Minimum versions are not pinned yet.

AI assistance (Claude Code) was used to prepare this branch. I have carefully reviewed the changes, tests, and results.
