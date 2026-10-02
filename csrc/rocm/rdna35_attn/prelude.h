// What every variant of a group unit shares, included before the first one
// so that the kernel's own includes find their guards set and add nothing to
// the variant namespaces.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "rdna35_attn/launch.h"
#include "rdna35_attn/macros.h"

#define RDNA35_VLLM 1
