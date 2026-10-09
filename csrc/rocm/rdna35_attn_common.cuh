// Primitives the RDNA3.5 decode and prefill kernels share: element types,
// LDS barrier, cross-lane moves, rounding and WMMA.
//
// Included by rdna35_decode_attn.cu and rdna35_prefill_attn.cu after they
// define kBf16, once per variant: a generated unit holds several variants,
// each in a namespace of its own, so this file has no include guard.

using elem_t = std::conditional_t<kBf16, __bf16, _Float16>;
typedef elem_t e16 __attribute__((ext_vector_type(16)));
typedef elem_t e8 __attribute__((ext_vector_type(8)));
typedef elem_t e2 __attribute__((ext_vector_type(2)));
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef short s16 __attribute__((ext_vector_type(16)));
typedef short s2v __attribute__((ext_vector_type(2)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned u8v __attribute__((ext_vector_type(8)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));
typedef unsigned u2v __attribute__((ext_vector_type(2)));

// Orders LDS only: __syncthreads() would also wait for every outstanding
// global load.
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// fetch-inactive is set so the compiler does not tie the destination to a
// copy of the source; every lane read is active anyway.
//
// The other 16-lane half's value: lane l <-> l ^ 16.
__device__ __forceinline__ unsigned xhalf_u(unsigned value) {
  return (unsigned)__builtin_amdgcn_permlanex16(
      (int)value, (int)value, 0x76543210u, 0xFEDCBA98u, true, false);
}

__device__ __forceinline__ float xhalf(float value) {
  return __builtin_bit_cast(float,
                            xhalf_u(__builtin_bit_cast(unsigned, value)));
}

// Within each 16-lane row: lane i takes lane i & 7's value, or lane i | 8's.
__device__ __forceinline__ unsigned lower8(unsigned value) {
  return (unsigned)__builtin_amdgcn_permlane16(
      (int)value, (int)value, 0x76543210u, 0x76543210u, true, false);
}

__device__ __forceinline__ float upper8f(float value) {
  const int bits = __builtin_bit_cast(int, value);
  return __builtin_bit_cast(
      float, __builtin_amdgcn_permlane16(bits, bits, 0xFEDCBA98u, 0xFEDCBA98u,
                                         true, false));
}

__device__ __forceinline__ float weight_of(float partMax, float maxAll) {
  return (maxAll == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(partMax - maxAll);
}

// Rounded to nearest even; a bf16 NaN may come out as Inf.
__device__ __forceinline__ elem_t to_elem(float x) {
  if constexpr (kBf16) {
    const unsigned bits = __builtin_bit_cast(unsigned, x);
    return __builtin_bit_cast(
        elem_t, (unsigned short)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16));
  } else {
    return (elem_t)x;
  }
}

__device__ __forceinline__ f8 wmma(e16 a, e16 b, f8 c) {
  if constexpr (kBf16)
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
        __builtin_bit_cast(s16, a), __builtin_bit_cast(s16, b), c);
  else
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
        __builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b), c);
}
