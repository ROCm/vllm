# SPDX-License-Identifier: Apache-2.0
"""A4W4 (mxfp4 weights x mxfp4 activations) MoE experts backed by aiter's
Triton `moe_gemm_a4w4` kernel.

Why this exists
---------------
On gfx1250 none of the pre-existing native A4W4 MoE paths work:

  * FlyDSL     -> LLVM backend crash legalising a 128-bit
                  `llvm.amdgcn.raw.ptr.buffer.load.lds` (aiter have confirmed
                  these kernels are not currently implemented).
  * CK 2-stage -> composable_kernel fails to compile
                  (`blockwise_gemm_pipeline_xdlops_..._v3.hpp`, non-constexpr
                  `if constexpr` condition).
  * Gluon      -> compiles and runs but returns NaN
                  (aiter's own test_moe_gemm_a4w4.py: 0/64 pass with
                  backend="gluon", 56/64 with backend="triton").

The Triton backend of the *same* `moe_gemm_a4w4` entry point is numerically
correct at DeepSeek-R1's shapes, so this class routes through it. `backend` is
read from an env var so switching to gluon is a one-liner once it is fixed.

Design notes (all verified empirically before this was written)
---------------------------------------------------------------
1. Weight layout: vLLM stores `[E, N, K]`; the kernel wants `[E, K, N]` AND
   asserts mxfp weights are column-major. A plain `.transpose(-1, -2)` view is
   exactly that, so NO repacking is needed. Forcing `.contiguous()` raises
   "`w` must be column-major when it has data-type mxfp".

2. Activation is applied OUTSIDE the kernel. `moe_gemm_a4w4(apply_swiglu=True)`
   is gpt-oss shaped: it reads gate/up INTERLEAVED (`[..., ::2]`/`[..., 1::2]`),
   clamps at `limit` (default 1.0) and multiplies by `(linear + 1)`. DeepSeek
   needs `silu(gate) * up` over contiguous halves, unclamped, no residual, and
   the kernel exposes no gate-mode switch. Doing it externally costs one kernel
   launch and avoids three silent numerical errors.

3. Routing is rebuilt from vLLM's precomputed `topk_ids`/`topk_weights`.
   aiter's `routing()` does support DeepSeek sigmoid grouped-top-k natively, but
   it consumes raw logits and vLLM has already routed by the time we are called.
   `sort_tokens` is used rather than `fused_routing_from_topk` (hard-capped at
   `n_tokens*topk <= 4096`, i.e. 512 tokens at topk=8) or
   `compute_expt_data_torch` (Python loop over every expert).
"""

import os

import torch
import torch.nn.functional as F
import triton

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
    kMxfp4Static,
)

# Backend for aiter's moe_gemm_a4w4: "gluon" or "triton".
#
# Default is "gluon", matching both aiter's own default on gfx1250
# (moe_gemm_a4w4 picks gluon when backend=None) and the aiter/triton team's
# guidance that gluon is the only A4W4 kernel they consider supported.
#
# "triton" is kept selectable: it passes more of aiter's unit tests on some
# stacks, but the team reports the triton a4w4 path is not fully implemented
# for gfx1250, and it produced wrong output end-to-end here. Neither backend
# has been observed working on this machine -- see _selftest_backend().
_A4W4_BACKEND = os.getenv("VLLM_ROCM_A4W4_TRITON_BACKEND", "gluon")

# Set VLLM_ROCM_A4W4_SKIP_SELFTEST=1 to bypass the startup numerical check.
# Only do this if you accept that a broken kernel will silently produce
# garbage output rather than failing.
_SKIP_SELFTEST = os.getenv("VLLM_ROCM_A4W4_SKIP_SELFTEST", "0").lower() in ("1", "true")

_SELFTEST_STATE = {"done": False}


def _as_u8(t: torch.Tensor) -> torch.Tensor:
    """Raw uint8 byte view of a 1-byte-per-element tensor.

    aiter's a4w4 kernel is written against the uint8 packing that
    `downcast_to_mxfp` emits; vLLM presents the same bytes as
    float4_e2m1fn_x2 / float8_e8m0fnu, which Triton cannot canonicalize.
    """
    if t.dtype == torch.uint8:
        return t
    assert t.element_size() == 1, (
        f"expected a 1-byte dtype for mxfp4 data, got {t.dtype}"
    )
    return t.view(torch.uint8)


def _build_routing(
    topk_weights: torch.Tensor, topk_ids: torch.Tensor, n_expts_tot: int
):
    """Build aiter RoutingData + gather/scatter indices from vLLM's top-k.

    Returns (routing_data, gather_indx, scatter_indx, gate_scal).
    """
    from aiter.ops.triton.moe.moe_routing.bitmatrix import Bitmatrix
    from aiter.ops.triton.moe.moe_routing.routing import (
        ExptData,
        RoutingData,
        sort_tokens,
        sort_tokens_fused,
    )

    n_tokens, n_expts_act = topk_ids.shape
    dev = topk_ids.device

    # block_m: aiter's own heuristic, reproduced verbatim from routing()
    m = n_tokens * n_expts_act
    tokens_per_expt = max(1, m // n_expts_tot)
    block_m = max(16, min(triton.next_power_of_2(tokens_per_expt), 128))

    if n_tokens <= 16:
        hist_block_m = triton.next_power_of_2(max(n_tokens, 1))
        sort_fn = sort_tokens_fused
    else:
        hist_block_m = 32
        sort_fn = sort_tokens

    # Bitmatrix: one bit per (token, expert). Storage layout, padding and both
    # scratchpads mirror aiter's topk() exactly -- the reduction kernel reads
    # scratchpad_partials directly and Bitmatrix.sum() does not allocate it.
    block_n = max(32, triton.next_power_of_2(n_expts_tot))
    block_s, tile_size = 128, 8
    n_cols_pad = triton.cdiv(n_expts_tot, block_n) * block_n
    n_cols_words = n_cols_pad // 32

    base = torch.zeros(
        (n_cols_words, triton.cdiv(n_tokens, 32) * 32),
        dtype=torch.int32,
        device=dev,
    )
    ids = topk_ids.to(torch.int64)
    vals = (torch.ones_like(ids) << (ids & 31)).to(torch.int32)
    base.transpose(0, 1)[:n_tokens].scatter_add_(1, ids >> 5, vals)
    bm_data = base.view(torch.uint32).transpose(0, 1)[:n_tokens]

    s_cols = triton.cdiv(n_cols_pad, block_s) * block_s
    scratchpad = torch.zeros((s_cols,), dtype=torch.int32, device=dev)
    pids_x = triton.cdiv(n_tokens, hist_block_m * tile_size)
    scratchpad_partials = torch.zeros(
        (n_cols_pad, pids_x * tile_size), dtype=torch.int32, device=dev
    ).transpose(0, 1)

    bitmatrix = Bitmatrix(
        bm_data,
        [n_tokens, n_expts_tot],
        scratchpad=scratchpad,
        scratchpad_partials=scratchpad_partials,
    )

    (
        hist,
        topk_indx,
        gate_indx,
        gate_scal,
        token_offs_raw,
        token_offs_pad,
        block_pid_map,
    ) = sort_fn(
        topk_weights.contiguous(),
        topk_ids.to(torch.int32).contiguous(),
        n_expts_tot,
        bitmatrix,
        block_m,
        hist_block_m,
    )
    expt_data = ExptData(hist, token_offs_raw, token_offs_pad, block_pid_map)
    routing_data = RoutingData(
        block_m=block_m,
        gate_scal=gate_scal,
        expt_hist=hist,
        n_expts_tot=n_expts_tot,
        n_expts_act=n_expts_act,
        expt_data=expt_data,
    )
    return routing_data, topk_indx, gate_indx, gate_scal


def _selftest_backend(backend: str) -> None:
    """One-off numerical check of the selected a4w4 kernel. Raises if broken.

    A broken a4w4 kernel does NOT crash -- it returns NaN or garbage, which
    propagates to exactly-uniform logits and a server that looks healthy while
    emitting nonsense (observed on this machine: every generated token decodes
    to id 0 with logprob -ln(vocab_size)). Failing loudly at startup is
    strictly better than serving that.

    Runs a small A4W4 MoE GEMM1 (with gather) through `backend` and compares
    against aiter's own `moe_gemm_torch` reference fed the *same* mxfp4-
    quantised inputs, so a correct kernel should agree to near machine
    precision. Costs one tiny kernel launch, once per process.
    """
    if _SELFTEST_STATE["done"] or _SKIP_SELFTEST:
        _SELFTEST_STATE["done"] = True
        return

    from aiter.ops.triton.moe.moe_op_gemm_a4w4 import (
        moe_gemm_a4w4,
        moe_gemm_torch,
        mxfp4_quant,
    )
    from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp, upcast_from_mxfp

    dev = torch.device("cuda")
    M, K, N, E, topk = 64, 512, 256, 8, 4
    g = torch.Generator(device=dev).manual_seed(0)

    x = torch.randn((M, K), dtype=torch.bfloat16, device=dev, generator=g) / 4

    # Build the weight in vLLM's STORAGE orientation [E, N, K] and quantise
    # along the contraction dim K -- exactly how the checkpoint stores mxfp4
    # (one e8m0 scale per 32-element K block, per output row N). apply() then
    # hands the kernel a transposed *view* ([E, K, N]) with a transposed scale;
    # mirror that here so the self-test exercises the real non-contiguous layout
    # and scale transpose, not a clean contiguous [E, K, N] the model never
    # produces. (A layout/scale mismatch is the leading suspect for the
    # prefill bug, and the previous self-test could not have caught it.)
    w_nk = torch.randn((E, N, K), dtype=torch.bfloat16, device=dev, generator=g) / 8
    w_q_nk, w_s_nk = downcast_to_mxfp(w_nk, torch.uint8, axis=-1)
    w_ref_nk = upcast_from_mxfp(w_q_nk, w_s_nk, torch.bfloat16, axis=-1)

    x_q, x_s = mxfp4_quant(x)
    x_ref = upcast_from_mxfp(x_q, x_s, torch.bfloat16, axis=-1)

    # topk via scores so expert ids are unique per row (the bitmatrix in
    # _build_routing sets one bit per (token, expert) and would miscount dupes)
    scores = torch.rand((M, E), device=dev, generator=g)
    tw, ti = torch.topk(scores, topk, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(torch.bfloat16).contiguous()
    ti = ti.to(torch.int32).contiguous()

    rdata, gather_indx, _scatter_indx, _gate_scal = _build_routing(tw, ti, E)

    # Transposed views are exactly what apply() feeds the kernel -- never
    # .contiguous() (the kernel asserts column-major mxfp weights). The
    # reference is our ground truth, so it may be made contiguous freely; only
    # the kernel input must stay a view for this to test the real path.
    w_q_t = _as_u8(w_q_nk).transpose(-1, -2)
    w_s_t = _as_u8(w_s_nk).transpose(-1, -2)
    w_ref_t = w_ref_nk.transpose(-1, -2).contiguous()

    ref = moe_gemm_torch(x_ref, w_ref_t, None, rdata, gather_indx, None, None, False)
    got = moe_gemm_a4w4(
        x_q,
        w_q_t,
        x_s,
        w_s_t,
        None,
        rdata,
        gather_indx=gather_indx,
        scatter_indx=None,
        swizzle_mx_scale=None,
        preshuffle_weights=False,
        apply_swiglu=False,
        backend=backend,
    )

    r, o = ref.float(), got.float()
    finite = bool(torch.isfinite(o).all())
    max_rel = (
        float(((o - r).abs() / r.abs().clamp_min(1e-3)).max())
        if finite
        else float("inf")
    )
    tol = 1e-2

    if not finite or max_rel > tol:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        raise RuntimeError(
            f"AiterTritonA4W4Experts self-test FAILED for backend='{backend}' "
            f"on {get_arch()}: finite={finite} max_rel_err={max_rel:.6g} "
            f"(tolerance {tol}).\n"
            f"The a4w4 MoE kernel is not numerically correct on this machine, so "
            f"serving would produce garbage output (uniform logits) rather than "
            f"crashing. Refusing to start.\n"
            f"Try the other backend via VLLM_ROCM_A4W4_TRITON_BACKEND="
            f"{'triton' if backend == 'gluon' else 'gluon'}, or reproduce "
            f"standalone with:\n"
            f"  pytest op_tests/triton_tests/moe/test_moe_gemm_a4w4.py -k {backend}\n"
            f"Set VLLM_ROCM_A4W4_SKIP_SELFTEST=1 to bypass (NOT recommended)."
        )

    _SELFTEST_STATE["done"] = True


class AiterTritonA4W4Experts(mk.FusedMoEExpertsModular):
    """MXFP4 weights x MXFP4 activations MoE via aiter's Triton a4w4 GEMM."""

    @property
    def expects_unquantized_inputs(self) -> bool:
        # We quantise hidden states to mxfp4 ourselves via mxfp4_quant.
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        from vllm.platforms.rocm import on_gfx1250

        return on_gfx1250()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        # W4A4 only. W4A16 stays on the existing paths.
        return (weight_key, activation_key) == (kMxfp4Static, kMxfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Only plain SiLU: the external silu_and_mul below assumes contiguous
        # gate/up halves, which is what DeepSeek uses.
        return activation in [MoEActivation.SILU]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # GEMM2 already applies routing weights (gammas) and reduces over topk
        # via scatter_indx, so the modular finalize step must be a no-op.
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Intermediates are allocated by the aiter kernels themselves.
        return (0,), (0,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        from aiter.ops.triton.moe.moe_op_gemm_a4w4 import moe_gemm_a4w4, mxfp4_quant

        # Fails the engine at init (during vLLM's profile run) rather than
        # letting a broken kernel serve silently-wrong output.
        _selftest_backend(_A4W4_BACKEND)

        if expert_map is not None:
            raise NotImplementedError(
                "AiterTritonA4W4Experts does not support expert parallelism "
                "(expert_map) yet."
            )

        w1_scale = self.quant_config.w1_scale
        w2_scale = self.quant_config.w2_scale
        assert w1_scale is not None and w2_scale is not None

        n_expts_tot = w1.shape[0]

        rdata, gather_indx, scatter_indx, gate_scal = _build_routing(
            topk_weights, topk_ids, n_expts_tot
        )

        # vLLM stores [E, N, K]; the kernel needs column-major [E, K, N].
        # These transposes are views -- no copy, and .contiguous() would break it
        # ("`w` must be column-major when it has data-type mxfp").
        #
        # _as_u8: vLLM hands these over as torch.float4_e2m1fn_x2 (and the
        # scales as float8_e8m0fnu), but aiter's kernel is written against the
        # raw uint8 byte view that downcast_to_mxfp produces, and Triton's
        # type_canonicalisation_dict has no entry for float4_e2m1fn_x2
        # (KeyError at launch). All of these are 1 byte per element, so the
        # view is free and shape-preserving. Must be taken BEFORE the transpose,
        # since .view() needs a contiguous trailing dim.
        w1_t, w1_s_t = _as_u8(w1).transpose(-1, -2), _as_u8(w1_scale).transpose(-1, -2)
        w2_t, w2_s_t = _as_u8(w2).transpose(-1, -2), _as_u8(w2_scale).transpose(-1, -2)

        # ---- GEMM1: gather tokens into expert-sorted order, no fused act ----
        xq, xs = mxfp4_quant(hidden_states)
        y1 = moe_gemm_a4w4(
            xq,
            w1_t,
            xs,
            w1_s_t,
            None,
            rdata,
            gather_indx=gather_indx,
            scatter_indx=None,
            swizzle_mx_scale=None,
            preshuffle_weights=False,
            apply_swiglu=False,
            backend=_A4W4_BACKEND,
        )

        # ---- activation: silu(gate) * up over contiguous halves ----
        half = y1.shape[-1] // 2
        act = (F.silu(y1[..., :half].float()) * y1[..., half:].float()).to(y1.dtype)

        # ---- GEMM2: routing weights + scatter-reduce back to [M, K] ----
        # If the router weights were already folded into the input, do not
        # apply them a second time here.
        gammas = None if apply_router_weight_on_input else gate_scal
        aq, as_ = mxfp4_quant(act)
        result = moe_gemm_a4w4(
            aq,
            w2_t,
            as_,
            w2_s_t,
            None,
            rdata,
            gather_indx=None,
            scatter_indx=scatter_indx,
            gammas=gammas,
            swizzle_mx_scale=None,
            preshuffle_weights=False,
            apply_swiglu=False,
            backend=_A4W4_BACKEND,
        )

        output.copy_(result.to(output.dtype))
