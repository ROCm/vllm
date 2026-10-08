#
# The RDNA3.5 decode-attention variants built into _rocm_C, for gfx1151.
#
# vllm/v1/attention/ops/rdna35_variants.csv is the one list of variants: its
# header names the kernel's compile-time defines, each row is one build.
# CMake re-runs whenever the list or the generator changes. At configure
# time csrc/rocm/generate_rdna35_attn.py writes one translation unit per
# (configuration, dtype) and the registry's table.
#
# Creates the object library ${TARGET} (the kernels, compiled for gfx1151
# alone, linked into _rocm_C) and sets ${OUT_INCLUDE} to the directory of
# rdna35_variants.inc, which the registry includes.
#
function(rdna35_attn_target TARGET OUT_INCLUDE)
  set(_CSV ${CMAKE_SOURCE_DIR}/vllm/v1/attention/ops/rdna35_variants.csv)
  set(_GEN ${CMAKE_SOURCE_DIR}/csrc/rocm/generate_rdna35_attn.py)
  set(_OUT ${CMAKE_CURRENT_BINARY_DIR}/rdna35_attn)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_CSV} ${_GEN})
  execute_process(
    COMMAND ${Python_EXECUTABLE} ${_GEN} ${_CSV} ${_OUT}
    OUTPUT_VARIABLE _UNITS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  string(REPLACE "\n" ";" _UNITS "${_UNITS}")

  add_library(${TARGET} OBJECT ${_UNITS})
  set_source_files_properties(${_UNITS} PROPERTIES LANGUAGE ${VLLM_GPU_LANG})
  set_target_properties(${TARGET} PROPERTIES
    ${VLLM_GPU_LANG}_ARCHITECTURES gfx1151
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/csrc/rocm)
  # The optimisation level the kernels were tuned at, whatever the build type;
  # -g0 keeps RelWithDebInfo from carrying debug info. The kernel keeps values
  # some variants leave unused rather than #if them out per knob, which
  # -Werror=unused-variable would refuse.
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:${VLLM_GPU_LANG}>:${VLLM_ROCM_EXT_FLAGS}>
    -O3 -g0 -Wno-unused-variable)
  set(${OUT_INCLUDE} ${_OUT} PARENT_SCOPE)
endfunction()
