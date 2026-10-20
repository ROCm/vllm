// The tables of builds: one for the archs with 1536 VGPRs a SIMD lane (gfx1151)
// and one for those with 1024 (gfx1150, gfx1152, gfx1153), whose best launch
// knobs differ. Each arch builds and serves its table.
#pragma once

#include "rdna_singleseq_attn_launch.h"
#include "rdna_singleseq_decode_configs.h"
#include "rdna_singleseq_decode_configs_1024vgpr.h"
#include "rdna_singleseq_prefill_configs.h"
#include "rdna_singleseq_prefill_configs_1024vgpr.h"

namespace rdna_singleseq {

struct Vgpr1536 {
  static constexpr const auto& decode = kDecodeConfigs;
  static constexpr const auto& prefill = kPrefillConfigs;
};

struct Vgpr1024 {
  static constexpr const auto& decode = vgpr1024::kDecodeConfigs;
  static constexpr const auto& prefill = vgpr1024::kPrefillConfigs;
};

// The launch function of each entry of Table::decode, null until its unit
// (cmake/rocm_rdna_singleseq_attn.cmake) registers it. Defined in
// rdna_singleseq_registry.cu.
template <class Table>
LaunchFn* decode_launchers();

// Likewise for Table::prefill.
template <class Table>
PrefillFn* prefill_launchers();

}  // namespace rdna_singleseq
