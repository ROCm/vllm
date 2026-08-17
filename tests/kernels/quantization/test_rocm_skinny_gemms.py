# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest
import torch

import vllm._custom_ops as ops
from tests.kernels.quant_utils import ref_dynamic_per_tensor_fp8_quant
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.utils import wvsplitkrc_dispatch
from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.utils.platform_utils import num_compute_units

# Global per-test cleanup costs more than the tests themselves.
# These tests can just cleanup once at the end
pytestmark = pytest.mark.skip_global_cleanup

SEEDS = [0]

FP32_EPS = torch.finfo(torch.float32).eps

# bias_mode: 0 = none, 1 = (m,), 2 = (n, m).
BIAS_MODES = [0, 1, 2]

# These options are independent in the kernels, so the sets below cover every
# pair of them instead of the full product. `scale` is the input scaling handed
# to gemm_inputs(). wvSplitKrc crosses the same axes in full instead, but only
# over the four shapes in NKM_REPS_WVSPLITKRC.
OPTIONS_WVSPLITK = [
    # dtype, padded_a, padded_b, bias_mode, scale
    (torch.float16, False, False, 0, "unit"),
    (torch.float16, False, True, 1, "he"),
    (torch.float16, True, False, 2, "he"),
    (torch.bfloat16, True, True, 0, "he"),
    (torch.bfloat16, True, False, 1, "unit"),
    (torch.bfloat16, False, True, 2, "unit"),
]

# Here `scale` rides on the quantization scales rather than the inputs; see
# test_rocm_wvsplitk_fp8_kernel.
OPTIONS_WVSPLITK_FP8 = [
    # dtype, padded_a, padded_b, biased, scale
    (torch.float16, False, False, True, "unit"),
    (torch.bfloat16, True, False, False, "unit"),
    (torch.bfloat16, False, True, True, "unit"),
    (torch.float16, True, False, True, "he"),
    (torch.float16, False, True, False, "he"),
    (torch.bfloat16, True, True, False, "he"),
]

DTYPES = [torch.bfloat16, torch.float16]

# Specific (N, K, M) combinations for targeted testing
NKM_FACTORS_LLMM1 = [
    # Small, medium, large cases
    (1, 8, 16),
    (1, 32, 64),
    (1, 128, 256),
    (1, 512, 1024),
    (1, 2048, 4096),
    # Edge cases with specific K sizes
    (1, 6144, 1024),
    (1, 8192, 2048),
    # Very large case
    (1, 4096, 8192),
]

NKM_FACTORS_WVSPLITK = [
    # Different batch sizes with key dimensions
    (1, 32, 16),
    (1, 64, 64),
    (2, 256, 256),
    (3, 1024, 1024),
    (4, 4096, 4096),
    (4, 4096, 4096 + 1),
    (4, 4096 + 16, 4096),
    (4, 4096 + 16, 4096 + 1),
    # Extended K values
    (1, 9216, 512),
    (2, 10240, 1024),
    (4, 16384, 8192),
    (4, 16384 * 2, 8192),
    (4, 16384 * 2, 8192 + 1),
    (4, 16384 * 2 + 16, 8192),
    (4, 16384 * 2 + 16, 8192 + 1),
    # Minimum M constraint validation (m >= 8)
    (1, 64, 8),
    (2, 128, 8),
    (4, 256, 8),
]

# N is bucketed up to its next ^2 (16/32/64/128) and the remainder is masked
N_FACTORS_WVSPLITKRC = [
    13,
    16,
    17,
    25,
    29,
    31,
    32,
    41,
    51,
    64,
    71,
    81,
    91,
    103,
    117,
    128,
]
# K shards are 512 wide, evenly divided or not, +8 for a partial 8-element load.
# The low K's dispatch with CHUNKK=2 and the high ones with CHUNKK=1;
# test_wvsplitkrc_chunkk2_is_covered pins that both sides stay non-empty.
K_FACTORS_WVSPLITKRC = [2560, 2560 + 8, 2880, 2880 + 8, 3072, 3072 + 8]
# M tiles are 64 rows, +16 for a partial tile
M_FACTORS_WVSPLITKRC = [128, 128 + 16, 256, 256 + 16, 640, 640 + 16]

# Shapes carrying the dtype/bias/padding/scale axes, which are independent of
# tiling: the GrpsShrB=1 path, an odd N with both K and M unaligned, a mid odd
# N, and the largest N and M that still dispatch here.
NKM_REPS_WVSPLITKRC = [
    (13, 2880, 128),
    (31, 2888, 144),
    (103, 3080, 256),
    (128, 3072, 640),
]

# (N, K, M) with more K-shards than the readback can stage in one LDS pass,
# over both N-tile counts.
NKM_FACTORS_WVSPLITKRC_LARGE_K = [
    (128, 6144, 128),
    (96, 8192, 128),
    (128, 12288, 128),
    (32, 12288, 128),
]

NKM_FACTORS_WVSPLITK_FP8 = [
    # FP8-specific cases with K % 16 == 0
    (1, 16, 16),
    (1, 32, 16 + 16),
    (1, 64, 64),
    (1, 64, 64 + 16),
    (1, 64 + 16, 64),
    (1, 64 + 16, 64 + 16),
    (4, 64, 64),
    (4, 64, 64 + 16),
    (4, 64 + 16, 64),
    (4, 64 + 16, 64 + 16),
    (2, 512, 512),
    (3, 512, 512),
    (3, 512, 512 + 16),
    (4, 512, 512),
    (3, 2048, 2048),
    (3, 2048, 2048 + 16),
    (4, 2048 + 16, 2048),
    (4, 2048 + 16, 2048 + 16),
    (4, 4096, 4096),
    (4, 16400, 2048),
    (4, 16400, 2048 + 16),
    # Extended FP8 dimensions not covered by WVSPLITK
    (1, 14336, 1024),
    (2, 24576, 2048),
    (4, 32768, 28672),
    (4, 32768 * 2, 28672),
    (4, 32768 * 2, 28672 + 16),
    (4, 32768 * 2 + 16, 28672),
    (4, 32768 * 2 + 16, 28672 + 16),
]


@pytest.fixture(scope="module", autouse=True)
def cleanup_after_all_tests():
    yield
    cleanup_dist_env_and_memory()


def pad_fp8(weight):
    num_pad = 256 // weight.element_size()
    import torch.nn.functional as F

    return F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]


def gemm_inputs(n, k, m, dtype, scale):
    """Activations at unit variance; only the weights carry fan-in scaling.

    Scaling both costs a factor sqrt(2/k) of output signal (~38x at k=3072),
    which any tolerance then has to be read against. `scale="unit"` leaves both
    unscaled, putting outputs at ~sqrt(k) to exercise large accumulators.
    """
    fan_in = math.sqrt(2 / k) if scale == "he" else 1.0
    A = torch.randn(n, k, dtype=dtype, device="cuda")
    B = torch.randn(m, k, dtype=dtype, device="cuda") * fan_in
    return A, B


def make_bias(bias_mode, n, m, dtype):
    if bias_mode == 1:
        return torch.rand(m, dtype=dtype, device="cuda") * 2 - 1
    if bias_mode == 2:
        return torch.rand(n, m, dtype=dtype, device="cuda") * 2 - 1
    return None


def fp8_abs(t):
    """|x| for e4m3 by clearing the sign bit, without an fp32 materialization."""
    return (t.contiguous().view(torch.uint8) & 0x7F).view(t.dtype)


def assert_gemm_close(out, ref, A, B, terms=None, slack=4.0):
    """Compare against a bound built from the magnitudes actually summed.

    Split-K reorders an fp32 accumulation, so its error tracks sum(|a_i b_i|)
    rather than |result|; dividing by the result instead makes the check
    vacuous wherever the dot product cancels, and scale-dependent everywhere
    else. The second term is the rounding to `out.dtype`, which the kernel and
    the reference apply independently and which usually dominates.

    `terms` overrides sum(|a_i b_i|) for inputs whose absolute values cannot be
    matmul'd directly, such as fp8.
    """
    eps = torch.finfo(out.dtype).eps
    k = A.shape[-1]
    if terms is None:
        terms = (A.abs() @ B.abs().t()).float()
    bound = slack * (eps * ref.float().abs() + FP32_EPS * math.sqrt(k) * terms)
    err = (out.float() - ref.float()).abs()
    worst = (err / bound).max().item()
    assert worst <= 1.0, (
        f"max |err|/bound = {worst:.3f}; max |err| = {err.max().item():.3e}, "
        f"median bound = {bound.median().item():.3e}"
    )


def run_wvsplitkrc(n, k, m, dtype, seed, padded_a, bias_mode, scale):
    torch.manual_seed(seed)
    cu_count = num_compute_units()
    if not wvsplitkrc_dispatch(n, k, m, cu_count)[1]:
        pytest.skip("Too large for wvSplitKrc")

    A, B = gemm_inputs(n, k, m, dtype, scale)
    BIAS = make_bias(bias_mode, n, m, dtype)
    if padded_a:
        A = pad_fp8(A)

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitKrc(A, B, cu_count, BIAS)

    assert_gemm_close(out, ref_out, A, B)


@pytest.mark.parametrize("n", N_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("k", K_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("m", M_FACTORS_WVSPLITKRC)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx950(), reason="only meant for gfx950")
def test_rocm_wvsplitkrc_shapes(n, k, m, seed):
    """Tiling and dispatch coverage over (N, K, M).

    The dtype/bias/padding/scale axes do not interact with tiling, so they are
    swept separately in test_rocm_wvsplitkrc_variants rather than crossed here.
    Unbiased is the sensitive setting to hold fixed: a bias makes the readback
    wait on its own loads, which masks a missing s_waitcnt on the LDS reads.
    """
    run_wvsplitkrc(n, k, m, torch.bfloat16, seed, False, 0, "he")


@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx950(), reason="only meant for gfx950")
def test_wvsplitkrc_chunkk2_is_covered():
    """Pin that the sweep above reaches both K-shard widths.

    CHUNKK=2 is admitted only for N_p2 > 16, under a CU budget, and under a
    shard cap -- narrow enough that a shape list can drift off it entirely
    without any other test failing.
    """
    cu_count = num_compute_units()
    dispatched = [
        wvsplitkrc_dispatch(n, k, m, cu_count)
        for n in N_FACTORS_WVSPLITKRC
        for k in K_FACTORS_WVSPLITKRC
        for m in M_FACTORS_WVSPLITKRC
    ]
    reached = {chunkk for chunkk, fits in dispatched if fits}
    assert reached == {1, 2}, f"sweep only reaches CHUNKK={sorted(reached)}"


@pytest.mark.parametrize("n,k,m", NKM_REPS_WVSPLITKRC)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.parametrize("padded_a", [False, True])
@pytest.mark.parametrize("scale", ["he", "unit"])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx950(), reason="only meant for gfx950")
def test_rocm_wvsplitkrc_variants(n, k, m, dtype, bias_mode, padded_a, scale, seed):
    """dtype, bias shape, A padding and input scale, at representative shapes."""
    run_wvsplitkrc(n, k, m, dtype, seed, padded_a, bias_mode, scale)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITKRC_LARGE_K)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("bias_mode", BIAS_MODES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@pytest.mark.skipif(not on_gfx950(), reason="only meant for gfx950")
def test_rocm_wvsplitkrc_large_k(n, k, m, dtype, bias_mode, seed):
    """K large enough that the split-K readback must stage LDS in batches.

    This is the only place bias meets that batched staging; the shapes in
    test_rocm_wvsplitkrc_variants all stage in a single pass.
    """
    run_wvsplitkrc(n, k, m, dtype, seed, False, bias_mode, "he")


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_LLMM1)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows_per_block", [2, 4, 8, 16])
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
@torch.inference_mode()
def test_rocm_llmm1_kernel(n, k, m, dtype, rows_per_block, seed):
    torch.manual_seed(seed)
    # TODO: Zero-centering the inputs causes errors for LLMM1!
    #      Without that the numbers quickly saturate, and may
    #      be giving false matches.
    A = torch.rand(n, k, dtype=dtype, device="cuda")
    B = torch.rand(m, k, dtype=dtype, device="cuda")

    ref_out = torch.matmul(A, B.t())
    out = ops.LLMM1(B, A, rows_per_block)

    torch.testing.assert_close(out, ref_out, atol=1e-8, rtol=1e-2)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK)
@pytest.mark.parametrize(
    "dtype,padded_a,padded_b,bias_mode,scale",
    OPTIONS_WVSPLITK,
)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(not current_platform.is_rocm(), reason="only test for rocm")
def test_rocm_wvsplitk_kernel(
    n, k, m, dtype, padded_a, padded_b, bias_mode, scale, seed
):
    torch.manual_seed(seed)
    cu_count = num_compute_units()

    A, B = gemm_inputs(n, k, m, dtype, scale)
    BIAS = make_bias(bias_mode, n, m, dtype)

    if padded_a:
        A = pad_fp8(A)
    if padded_b:
        B = pad_fp8(B)

    ref_out = torch.nn.functional.linear(A, B, BIAS)
    out = ops.wvSplitK(B, A.view(-1, A.size(-1)), cu_count, BIAS)

    assert_gemm_close(out, ref_out, A, B)


@pytest.mark.parametrize("n,k,m", NKM_FACTORS_WVSPLITK_FP8)
@pytest.mark.parametrize(
    "dtype,padded_a,padded_b,biased,scale",
    OPTIONS_WVSPLITK_FP8,
)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.skipif(
    not (current_platform.is_rocm() and current_platform.supports_fp8()),
    reason="only test for rocm fp8",
)
def test_rocm_wvsplitk_fp8_kernel(
    n, k, m, dtype, padded_a, padded_b, biased, scale, seed
):
    """The scale axis rides on the quantization scales, not on the inputs.

    Per-tensor dynamic quantization divides by amax, so scaling the inputs
    cancels out of the fp8 operands exactly and would leave this test
    unchanged. Scaling `scale_a`/`scale_b` keeps the operands bit-identical
    and moves only the output, which is the part that must move: the error
    bound grows as sqrt(k) * sum(|a_i b_i|), so at large K it swamps an O(1)
    bias unless the output is brought back onto the bias's scale.
    """
    torch.manual_seed(seed)

    A = torch.rand(n, k, device="cuda") * 2 - 1
    B = torch.rand(m, k, device="cuda") * 2 - 1

    A, scale_a = ref_dynamic_per_tensor_fp8_quant(A)
    B, scale_b = ref_dynamic_per_tensor_fp8_quant(B)
    if scale == "he":
        fan_in = math.sqrt(2 / k)
        scale_a, scale_b = scale_a * fan_in, scale_b * fan_in
    # Padding only restrides, so the term magnitudes are the same either way.
    terms = torch._scaled_mm(
        fp8_abs(A),
        fp8_abs(B).t(),
        out_dtype=torch.float32,
        scale_a=scale_a,
        scale_b=scale_b,
    )
    if padded_b:
        B = pad_fp8(B)
    if padded_a:
        A = pad_fp8(A)

    BIAS = None if (not biased) else (torch.rand(m, dtype=dtype, device="cuda") * 2 - 1)

    ref_out = torch._scaled_mm(
        A, B.t(), out_dtype=dtype, scale_a=scale_a, scale_b=scale_b, bias=BIAS
    )
    out = ops.wvSplitKQ(B, A, dtype, scale_a, scale_b, num_compute_units(), BIAS)

    assert_gemm_close(out, ref_out, A, B, terms=terms)

    if BIAS is not None and scale == "he":
        # A kernel that ignored BIAS would return exactly the unbiased
        # reference, so the check above tests the bias only where that
        # reference fails it. Pin that it does.
        ref_nobias = torch._scaled_mm(
            A, B.t(), out_dtype=dtype, scale_a=scale_a, scale_b=scale_b
        )
        with pytest.raises(AssertionError):
            assert_gemm_close(ref_nobias, ref_out, A, B, terms=terms)
