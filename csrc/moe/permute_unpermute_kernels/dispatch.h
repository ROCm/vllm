#pragma once

#ifdef USE_ROCM
  #include <hip/hip_bfloat16.h>
  #include <hip/hip_fp8.h>
#else
  #include <cuda_fp8.h>
#endif

#define MOE_SWITCH(TYPE, ...)                                     \
  at::ScalarType _st = ::detail::scalar_type(TYPE);               \
  switch (_st) {                                                  \
    __VA_ARGS__                                                   \
    default:                                                      \
      TORCH_CHECK(false, "[moe permute]data type dispatch fail!") \
  }

#define MOE_DISPATCH_CASE(enum_type, ...)                  \
  case enum_type: {                                        \
    using scalar_t = ScalarType2CudaType<enum_type>::type; \
    __VA_ARGS__();                                         \
    break;                                                 \
  }
#define MOE_DISPATCH_FLOAT_CASE(...)                            \
  MOE_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)         \
  MOE_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)          \
  MOE_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)      \
  MOE_DISPATCH_CASE(at::ScalarType::Float8_e5m2, __VA_ARGS__)   \
  MOE_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__) \
  MOE_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)

#define MOE_DISPATCH(TYPE, ...) \
  MOE_SWITCH(TYPE, MOE_DISPATCH_FLOAT_CASE(__VA_ARGS__))

template <at::ScalarType type>
struct ScalarType2CudaType;

template <>
struct ScalarType2CudaType<at::ScalarType::Float> {
  using type = float;
};
template <>
struct ScalarType2CudaType<at::ScalarType::Half> {
  using type = half;
};
template <>
struct ScalarType2CudaType<at::ScalarType::BFloat16> {
#ifdef USE_ROCM
  using type = hip_bfloat16;
#else
  using type = __nv_bfloat16;
#endif
};
template <>
struct ScalarType2CudaType<at::ScalarType::Byte> {
  using type = uint8_t;
};
template <>
struct ScalarType2CudaType<at::ScalarType::Float8_e5m2> {
#ifdef USE_ROCM
  using type = __hip_fp8_e5m2;
#else
  using type = __nv_fp8_e5m2;
#endif
};
template <>
struct ScalarType2CudaType<at::ScalarType::Float8_e4m3fn> {
#ifdef USE_ROCM
  using type = __hip_fp8_e4m3;
#else
  using type = __nv_fp8_e4m3;
#endif
};
