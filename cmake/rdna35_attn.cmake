#
# Sources of the RDNA3.5 decode-attention variants built into _rocm_C.
#
# csrc/rocm/rdna35_attn/variants.def is the one list of variants.  At
# configure time it is split by group into one generated unit per group,
# each rewritten only when its lines change, and CMake re-runs whenever the
# list changes.  So editing one group's lines rebuilds that group's unit and
# the registry (which includes the whole list), and leaves every other unit
# untouched.
#
# Sets ${OUT_SRCS} to the registry plus the generated units.
#
function(rdna35_attn_sources OUT_SRCS)
  set(_DIR ${CMAKE_SOURCE_DIR}/csrc/rocm/rdna35_attn)
  set(_DEF ${_DIR}/variants.def)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_DEF})

  file(STRINGS ${_DEF} _LINES REGEX "^RDNA35_VARIANT\\(")
  set(_GROUPS)
  foreach(_LINE IN LISTS _LINES)
    if(NOT _LINE MATCHES "^RDNA35_VARIANT\\(([A-Za-z0-9_]+),")
      message(FATAL_ERROR "variants.def: no group in '${_LINE}'")
    endif()
    set(_G ${CMAKE_MATCH_1})
    list(APPEND _GROUPS ${_G})
    string(APPEND _BODY_${_G}
      "#define RDNA35_V ${_LINE}\n"
      "#include \"rdna35_attn/instance.h\"\n"
      "#undef RDNA35_V\n")
  endforeach()
  list(REMOVE_DUPLICATES _GROUPS)
  list(LENGTH _LINES _NV)
  list(LENGTH _GROUPS _NG)
  message(STATUS "RDNA35 decode attention: ${_NV} variants in ${_NG} units")

  set(_OUT ${CMAKE_CURRENT_BINARY_DIR}/rdna35_attn)
  set(_SRCS ${_DIR}/registry.hip)
  foreach(_G IN LISTS _GROUPS)
    # file(CONFIGURE) leaves the file, and its timestamp, alone when the
    # content is unchanged: that is what keeps the other groups built.
    file(CONFIGURE OUTPUT ${_OUT}/${_G}.hip
      CONTENT "// Generated from variants.def, group ${_G}.  Do not edit.\n${_BODY_${_G}}"
      @ONLY)
    list(APPEND _SRCS ${_OUT}/${_G}.hip)
  endforeach()

  set_source_files_properties(${_SRCS} PROPERTIES
    INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/csrc/rocm
    # The optimisation level the kernels were tuned at, whatever the build
    # type; -g0 keeps RelWithDebInfo from carrying ~1000 kernels' debug info.
    # The kernel keeps values some variants leave unused rather than #if them
    # out per knob, which _rocm_C's -Werror=unused-variable would refuse.
    COMPILE_OPTIONS "-O3;-g0;-Wno-unused-variable")
  set(${OUT_SRCS} ${_SRCS} PARENT_SCOPE)
endfunction()
