// One decode-attention variant, built into _rocm_C.
//
// Included once per variant by the generated group units, after
//
//   #define RDNA35_V RDNA35_VARIANT(<a line of variants.def>)
//
// No include guard: every inclusion is another variant.  The kernel's defines
// read the entry's fields, its code goes into a namespace named after the
// entry, and rdna35::launch_<id> -- declared for the registry by variants.h
// -- launches it.  Everything defined here is undefined at the end, the
// kernel's own definitions by the kernel (RDNA35_VLLM).
#include "rdna35_attn/prelude.h"

#define RDNA35_VARIANT(...) (__VA_ARGS__)

#define HEAD_DIM RDNA35_FIELD(1, RDNA35_V)
#define NUM_Q_HEADS RDNA35_FIELD(2, RDNA35_V)
#define NUM_KV_HEADS RDNA35_FIELD(3, RDNA35_V)
#define MAXM RDNA35_FIELD(4, RDNA35_V)
#define BS RDNA35_FIELD(5, RDNA35_V)
#define LAYOUT RDNA35_FIELD(6, RDNA35_V)
#define NSEG RDNA35_FIELD(7, RDNA35_V)
#define RG RDNA35_FIELD(8, RDNA35_V)
#define MINB RDNA35_FIELD(9, RDNA35_V)
#define NW RDNA35_FIELD(10, RDNA35_V)
#define DSPL RDNA35_FIELD(11, RDNA35_V)
#define RSPL RDNA35_FIELD(12, RDNA35_V)
#define PF RDNA35_FIELD(13, RDNA35_V)
#define CPUB RDNA35_FIELD(14, RDNA35_V)
#define VINLDS RDNA35_FIELD(15, RDNA35_V)
#define BATCH RDNA35_FIELD(16, RDNA35_V)
#define DOT RDNA35_FIELD(17, RDNA35_V)
#define BFLY RDNA35_FIELD(18, RDNA35_V)
#define GT RDNA35_FIELD(19, RDNA35_V)
#define WIN RDNA35_FIELD(20, RDNA35_V)
#define KV_BF16 RDNA35_FIELD(21, RDNA35_V)
#define MUTATE 0
#define ABLATE 0
#define TIMING 0

namespace rdna35 {
namespace RDNA35_CALL(RDNA35_ID, RDNA35_PREPEND(v, RDNA35_V)) {
#include "rdna35_decode_attn.cu"
}  // namespace RDNA35_CALL(RDNA35_ID,RDNA35_PREPEND(v,RDNA35_V))

void RDNA35_CALL(RDNA35_ID,
                 RDNA35_PREPEND(launch, RDNA35_V))(const LaunchArgs& a) {
  RDNA35_CALL(RDNA35_ID, RDNA35_PREPEND(v, RDNA35_V))::launch(a);
}
}  // namespace rdna35

#undef RDNA35_VARIANT
#undef HEAD_DIM
#undef NUM_Q_HEADS
#undef NUM_KV_HEADS
#undef MAXM
#undef BS
#undef LAYOUT
#undef NSEG
#undef RG
#undef MINB
#undef NW
#undef DSPL
#undef RSPL
#undef PF
#undef CPUB
#undef VINLDS
#undef BATCH
#undef DOT
#undef BFLY
#undef GT
#undef WIN
#undef KV_BF16
#undef MUTATE
#undef ABLATE
#undef TIMING
#undef BTARGET
