#pragma once

#include <array>
#include <string_view>

// list of supported ROCm architecture names defined by the build scripts
//
// N.B. internal linkage (not `inline`) because VLLM_ROCM_ARCH_LIST could
// theoretically be defined differently when compiling different translation
// units -- though that isn't advised
constexpr auto kRocmArchs =
    std::to_array<std::string_view>({VLLM_ROCM_ARCH_LIST});

// Return true when `device_arch` identifies a specific gfx____ target or
// family of them.
template <unsigned... VERSIONS>
constexpr bool is_gfx(std::string_view device_arch) {
  static_assert(sizeof...(VERSIONS) > 0, "is_gfx needs a gfx version");
  constexpr auto prefix = [](unsigned version) -> std::string_view {
    switch (version) {
      case 11:
        return "gfx11";
      case 12:
        return "gfx12";
      default:
        return {};
    }
  };
  static_assert(((!prefix(VERSIONS).empty()) && ...),
                "unsupported gfx version");
  return (device_arch.starts_with(prefix(VERSIONS)) || ...);
}

static_assert(!is_gfx<11>("foo"));
static_assert(!is_gfx<12>("foo"));

static_assert(is_gfx<11>("gfx1100"));
static_assert(is_gfx<11>("gfx1151"));
static_assert(is_gfx<11, 12>("gfx1152"));
static_assert(!is_gfx<12>("gfx1153"));

static_assert(is_gfx<12>("gfx1200"));
static_assert(is_gfx<12>("gfx1201"));
static_assert(is_gfx<11, 12>("gfx1202"));
static_assert(!is_gfx<11>("gfx1250"));
