// Every decode-attention variant the wheel carries, from variants.def.
//
// Only the registry includes this: a group unit sees its own entries alone
// (cmake/rdna35_attn.cmake), so editing one group's lines of variants.def
// rebuilds that unit and the registry, not the rest.
#pragma once

#include "rdna35_attn/launch.h"
#include "rdna35_attn/macros.h"

namespace rdna35 {

#define RDNA35_VARIANT(...) \
  void RDNA35_CALL(RDNA35_ID, (launch, __VA_ARGS__))(const LaunchArgs&);
#include "rdna35_attn/variants.def"
#undef RDNA35_VARIANT

struct Variant {
  int key[RDNA35_NFIELDS];
  LaunchFn fn;
};

inline constexpr Variant kVariants[] = {
#define RDNA35_VARIANT(...) \
  {{RDNA35_KEY(__VA_ARGS__)}, &RDNA35_CALL(RDNA35_ID, (launch, __VA_ARGS__))},
#include "rdna35_attn/variants.def"
#undef RDNA35_VARIANT
};

}  // namespace rdna35
