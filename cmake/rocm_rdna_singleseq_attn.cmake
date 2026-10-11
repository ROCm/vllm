#
# Builds every row of
# vllm/v1/attention/ops/rocm_rdna_singleseq_{decode,prefill}_variants.csv into
# the object library ${TARGET} for ${ARCHES}, and sets ${OUT_INCLUDE} to the
# directory of the generated registry tables.
#
function(rdna_singleseq_attn_target TARGET ARCHES OUT_INCLUDE)
  set(_OPS ${CMAKE_SOURCE_DIR}/vllm/v1/attention/ops)
  set(_GEN ${CMAKE_SOURCE_DIR}/csrc/rocm/generate_rdna_singleseq_attn.py)
  set(_OUT ${CMAKE_CURRENT_BINARY_DIR}/rdna_singleseq_attn)
  set(_UNITS "")
  foreach(_KIND_CSV "decode:rocm_rdna_singleseq_decode_variants.csv"
                    "prefill:rocm_rdna_singleseq_prefill_variants.csv")
    string(REPLACE ":" ";" _KIND_CSV "${_KIND_CSV}")
    list(GET _KIND_CSV 0 _KIND)
    list(GET _KIND_CSV 1 _CSV)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_OPS}/${_CSV})
    execute_process(
      COMMAND ${Python_EXECUTABLE} ${_GEN} ${_OPS}/${_CSV} ${_OUT} --kind ${_KIND}
      OUTPUT_VARIABLE _KIND_UNITS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "\n" ";" _KIND_UNITS "${_KIND_UNITS}")
    list(APPEND _UNITS ${_KIND_UNITS})
  endforeach()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_GEN})

  add_library(${TARGET} OBJECT ${_UNITS})
  set_source_files_properties(${_UNITS} PROPERTIES LANGUAGE ${VLLM_GPU_LANG})
  set_target_properties(${TARGET} PROPERTIES
    ${VLLM_GPU_LANG}_ARCHITECTURES "${ARCHES}"
    POSITION_INDEPENDENT_CODE ON)
  target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/csrc/rocm)
  # -O3 regardless of build type. Some variants leave values unused.
  target_compile_options(${TARGET} PRIVATE
    $<$<COMPILE_LANGUAGE:${VLLM_GPU_LANG}>:${VLLM_ROCM_EXT_FLAGS}>
    -O3 -g0 -Wno-unused-variable)
  set(${OUT_INCLUDE} ${_OUT} PARENT_SCOPE)
endfunction()
