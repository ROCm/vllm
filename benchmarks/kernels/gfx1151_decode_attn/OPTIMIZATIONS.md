# Optimization repertoire

One entry per optimisation we land or reject, newest last. Each carries the
motivation from *our* kernel (not a general lesson), the C++ that changed, the
ISA before and after, and what it measured. The ISA is the point: this project
has a documented history of plausible ideas that lost, so an entry without a
before/after disassembly and a number is not an entry.

Entries are numbered and never renumbered. A rejected idea keeps its entry --
knowing what lost is why `reports/` exists.

---

## 001 — Split the score butterfly across both cross-lane pipes

**Status:** implemented behind `BFLY`, default still `0`. `BFLY=3` recommended.

### Motivation, from our case

The kernel is memory-bound at both `M`, but at `M=4` it stops *reaching* the
bus. Measured at `Hq=16/Hkv=2/S=32768`: `M=1` sits at 91 % of the KV roofline,
232-243 GB/s against a 247 GB/s ceiling, while `M=4` at `D=512` managed only
129.6 GB/s -- 52 % of the same ceiling, for KV traffic that is no larger.

The extra query tokens do not add KV bytes. What they add is work on the
critical path between the loads, so the wave issues its next load later and the
bus goes idle waiting for it. Shortening that path is therefore a bandwidth
optimisation, not an arithmetic one.

The ISA said where the time went. From `M=1` to `M=4` at fixed `MSPLIT`:

| | M=1 | M=4 | ratio |
| --- | --- | --- | --- |
| `global_load_b128` | 11 | 14 | 1.27x |
| `ds_bpermute_b32` | 20 | 80 | 4.00x |
| `s_waitcnt` | 31 | 110 | 3.55x |
| wall time | | | 1.87x |

KV streaming is flat -- the GQA reuse already works -- while the score
butterfly and its waits scale with `M`. The butterfly is the cost.

The obvious fix was refuted before we started: lowering the butterfly to DPP
measured 18 % slower across the sweep, because DPP is a *modifier fused onto
the add*, and the ISA forbids DPP inside VOPD. It converts adds that were
dual-issuing into adds that cannot.

`V_PERMLANE16` does not have that property. It is a standalone move, so the
adds stay plain `V_ADD_F32` and remain VOPD-eligible. DPP losing did not imply
permlane loses -- they fail differently, and only one had been measured.

But all-permlane is not the answer either: it moves every butterfly element
onto the VALU, which is the busy pipe, and regressed 8 % at `D=512, S=32768`.
The two mechanisms sit on *different pipes*, so the real win is using both at
once.

### What it does

`BFLY` routes each butterfly element to one of the two cross-lane pipes:
`ds_bpermute` on the LDS hardware, or `v_permlane16` / `v_permlanex16` on the
VALU. It is a ratio in quarters -- `0` sends everything to the LDS pipe, `4`
everything to the VALU, `3` sends three of every four elements to the VALU.

The split is over **elements, not strides**. The five strides are a dependence
chain (stride 2 consumes stride 1), so moving whole strides to the other pipe
buys no overlap; the `KPWE*MPW` elements *within* one stride are independent
and can occupy both pipes simultaneously.

### The C++ change

```c
// Before: every element on the LDS pipe.
for (int st = 1; st < LPR; st <<= 1) {
  const int addr = (lane ^ st) << 2;
  for (int c = 0; c < KPWE; ++c)
    for (int t = 0; t < MPW; ++t)
      s[c][t] = __builtin_bit_cast(
                    float, __builtin_amdgcn_ds_bpermute(
                               addr, __builtin_bit_cast(int, s[c][t]))) +
                s[c][t];
}
```

```c
// After: the element index picks the pipe.
#define BFLY_ON_VALU(idx) (((idx) & 3) < BFLY)
// Nibble i of the (lo, hi) pair is the source lane for destination lane i
// inside each 16-lane row, so the pair encodes an XOR-by-st swizzle.
#define BFLY_LO(st) ((st) == 1 ? 0x67452301u : (st) == 2 ? 0x54761032u \
                   : (st) == 4 ? 0x32107654u : 0xFEDCBA98u)
#define BFLY_HI(st) ((st) == 1 ? 0xEFCDAB89u : (st) == 2 ? 0xDCFE98BAu \
                   : (st) == 4 ? 0xBA98FEDCu : 0x76543210u)
// st == 16 is the only stride that leaves the 16-lane row.
#define BFLY_VALU(st, v)                                                  \
  ((st) >= 16 ? __builtin_amdgcn_permlanex16(                             \
                    __builtin_bit_cast(int, v),                           \
                    __builtin_bit_cast(int, v),                           \
                    0x76543210u, 0xFEDCBA98u, false, false)               \
              : __builtin_amdgcn_permlane16(                              \
                    __builtin_bit_cast(int, v),                           \
                    __builtin_bit_cast(int, v),                           \
                    BFLY_LO(st), BFLY_HI(st), false, false))
#define BFLY_XOR(st, addr, idx, v)                                        \
  (BFLY_ON_VALU(idx) ? BFLY_VALU(st, v)                                   \
                     : __builtin_amdgcn_ds_bpermute(                      \
                           addr, __builtin_bit_cast(int, v)))

for (int st = 1; st < LPR; st <<= 1) {
  const int addr = (lane ^ st) << 2;
  for (int c = 0; c < KPWE; ++c)
    for (int t = 0; t < MPW; ++t)
      s[c][t] = __builtin_bit_cast(
                    float, BFLY_XOR(st, addr, c * MPW + t, s[c][t])) +
                s[c][t];
}
```

Note `__shfl_xor` is not a substitute. For strides below 16 it lowers to DPP,
walking straight back into the refuted case; and it cannot prove the partner
index is in range, so it emits a `v_cmp_gt_u32` plus `v_cndmask` per stride.

### ISA, before and after

`-DHEAD_DIM=256 -DNUM_Q_HEADS=16 -DNUM_KV_HEADS=2 -DMAXM=4 -DMSPLIT=1 -DNSEG=2`

| BFLY | instructions | `ds_bpermute` | `permlane` | `s_waitcnt` | `v_dual` |
| --- | --- | --- | --- | --- | --- |
| 0 | 1234 | 80 | 0 | 110 | 88 |
| 1 | 1228 | 60 | 20 | 91 | 100 |
| 2 | 1272 | 40 | 40 | 75 | 98 |
| 3 | 1282 | 20 | 60 | **63** | **103** |
| 4 | 1280 | 0 | 80 | 46 | 98 |

**Before** (`BFLY=0`) -- every add is stalled behind an LDS return and issues
alone:

```asm
ds_bpermute_b32 v105, v84, v98
ds_bpermute_b32 v107, v84, v100
ds_bpermute_b32 v106, v84, v99
ds_bpermute_b32 v108, v84, v102
ds_bpermute_b32 v70,  v84, v80
s_waitcnt lgkmcnt(4)
v_add_f32_e32   v67, v98, v105
ds_bpermute_b32 v109, v84, v103
s_waitcnt lgkmcnt(4)
v_add_f32_e32   v98, v100, v107
```

**After** (`BFLY=3`) -- no `lgkmcnt` on this path, and the butterfly add now
dual-issues *with a dot product*:

```asm
v_permlane16_b32 v65, v65, s30, 0xefcdab89
v_permlane16_b32 v71, v71, s30, 0xefcdab89
v_permlane16_b32 v72, v72, s30, 0xefcdab89
v_permlane16_b32 v66, v66, s30, 0xefcdab89
v_dual_add_f32   v65, v98, v65 :: v_dual_dot2acc_f32_f16 v70, v24, v68
v_permlane16_b32 v67, v67, s30, 0xefcdab89
v_dual_add_f32   v71, v103, v71 :: v_dual_add_f32 v72, v77, v72
v_add_f32_e32    v67, v100, v67
```

That `v_dual_add_f32 :: v_dual_dot2acc_f32_f16` is the whole mechanism: the
reduction add rides along with the dot stream instead of blocking on LDS.

### Measured

Microseconds. `--reps 9` at S=128 and 512, `--reps 7` above. Lower is better;
**bold** is best per column.

`Hq=16 Hkv=2 D=512 M=4`

| BFLY | 128 | 512 | 1024 | 8192 | 16384 | 32768 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 10.29 | 22.26 | 38.15 | 265.17 | 521.40 | 1035.81 |
| 1 | 9.78 | 20.35 | 34.36 | 241.23 | 474.68 | 924.82 |
| 2 | 9.62 | 19.67 | 32.98 | 228.36 | 443.44 | 882.57 |
| 3 | 9.11 | 17.73 | 29.20 | 209.50 | **406.49** | **872.10** |
| 4 | **8.84** | **17.04** | **28.39** | **208.48** | 425.15 | 1117.66 |

`Hq=16 Hkv=2 D=256 M=4`

| BFLY | 128 | 512 | 1024 | 8192 | 32768 |
| --- | --- | --- | --- | --- | --- |
| 0 | 6.80 | 13.35 | 21.84 | 146.34 | 551.90 |
| 1 | 6.55 | 12.54 | 20.49 | 136.75 | 525.02 |
| 2 | 6.61 | 12.89 | 21.25 | 141.13 | 543.04 |
| 3 | 6.37 | 11.88 | 19.34 | **127.74** | 507.44 |
| 4 | **6.24** | **11.54** | **18.90** | 129.11 | **494.23** |

`Hq=32 Hkv=8 D=128 M=1`

| BFLY | 128 | 512 | 1024 | 8192 | 32768 |
| --- | --- | --- | --- | --- | --- |
| 0 | 5.78 | 15.03 | 27.62 | 210.10 | 824.02 |
| 1 | 5.78 | 15.00 | 27.52 | 209.62 | 821.41 |
| 2 | 5.66 | 14.35 | 26.18 | **198.44** | 775.62 |
| 3 | **5.65** | 14.35 | **26.17** | 198.60 | **775.52** |
| 4 | 5.66 | **14.34** | 26.18 | 198.56 | 775.97 |

`BFLY=3` against `BFLY=0`, across all 16 cells measured: **2.2 % to 23.5 %
faster, no regression at any context.** The win grows with context up to 8-16k
and with `M`, and is smallest at S=128 where fixed cost dominates.

`BFLY=4` is faster than `BFLY=3` at every short and mid context -- by about 3 %
at S=128 and S=512 -- but regresses 7.9 % at `D=512, S=32768`. Since one value
must serve the whole context range, that 3 % is the premium paid for the 22 %
at 32k, which is why the recommendation is 3 and not 4.

Gains saturate at `BFLY=2` for `D=128/M=1` and keep climbing to `3` for the
`M=4` shapes -- consistent with the butterfly being a larger share of the
critical path as `M` grows.

### Correctness

`max_rel` is identical to four significant digits against `BFLY=0` across all
contexts and both layouts (4.852e-04, 4.848e-04, 4.831e-04, 4.808e-04) -- the
butterflies are numerically equivalent, not merely both under tolerance. The
`--mutate 1` negative control fires at 2.842e+01.

### Rejected follow-up: `bound_ctrl` on the permlane

`__builtin_amdgcn_permlane16(old, src, lo, hi, fi, bound_ctrl)` with
`bound_ctrl=false` keeps `old` for lanes whose source is out of range, which
makes the instruction a read-modify-write and forces the allocator to hold
`old` live. Every one of our 16 selectors is in range, so `bound_ctrl` is
semantically a no-op here and `true` frees the allocator:

| | instructions | `v_mov_b32` | dual words | paired ops |
| --- | --- | --- | --- | --- |
| `bound_ctrl=false` | 1282 | 89 | 103 | 22.5 % |
| `bound_ctrl=true` | **1203** | **60** | 78 | 18.2 % |

79 fewer instructions and 29 fewer register copies -- and **no runtime
improvement**. Measured at `Hq=16/Hkv=2/D=512/M=4`, `BFLY=3`, reps=7:
29.01 / 200.12 / 409.93 / 861.83 against 29.20 / 209.50 / 406.49 / 872.10.
Three of the four deltas (-0.65 %, +0.85 %, -1.18 %) are below the 1.4 % p90
noise floor, so their signs carry no information; only the 8192 cell (-4.5 %)
clears it, on a single sample. VGPR went 108 -> 109 and pairing 22.5 % ->
18.2 %, both slightly worse.

**Reverted.** The only thing it demonstrably bought was a shorter instruction
stream, which is the third time on this kernel that instruction count has
failed to predict time. `bound_ctrl` stays `false`.

### What is still left in the generated code

Not optimal. Measured on the `BFLY=3` build:

- **VOPD pairing is 22 % of VALU ops** (917 ops in 814 instruction words;
  perfect pairing would be 459). About 80 of those ops are `permlane` and
  `ds_bpermute`, which are absent from the OPX/OPY lists and can never pair,
  but the rest could in principle. The wiki's `SRC2` read-port ceiling caps an
  accumulator-heavy stream near 78-84 %, so full pairing is not reachable --
  the gap is still wide.
- **60-89 `v_mov_b32`**, 5-7 % of the stream, pure register shuffling.
- **Every `permlane` carries a 32-bit literal.** VOP3 allows one SGPR plus one
  literal and the compiler already uses both slots, so this looks forced rather
  than missed.
- **53 `lgkmcnt` waits for only 20 `ds_bpermute`**, suggesting the waits are
  more conservative than the remaining LDS traffic requires.

The larger headroom is structural, not peephole. `BFLY=3` moved
`Hq=16/Hkv=2/D=512/M=4` at 32k from 129.6 to 155.8 GB/s, 52 % -> 63 % of the
247 GB/s ceiling, so roughly a third of the bus is still idle at `M=4`. And
`D=512/M=4` is forced onto `MSPLIT=2`, which reads KV twice, so part of that
remaining traffic is duplicate work rather than useful bytes -- relieving the
LDS pressure that forces `MSPLIT=2` is the next bandwidth lever, ahead of any
further work on the butterfly.

At `M=1` the kernel is at 91-98 % of the KV roofline, which is the correct
ceiling, so there is almost nothing left to win there.

### Open

`BFLY` is compile-time per *variant*, and variants are keyed on shape and `M`,
not on context -- one value must serve S=128 through S=32768. `BFLY=3` is the
only value measured that never regresses, which is why it is the recommendation
over the per-context optimum.

Not yet measured: `D=64`, `Hkv=1` shapes, and the full shape table. The default
stays `0` until the matrix is re-run.

---

## 002 — `__builtin_assume` on the two runtime scalars

**Status:** landed, unconditional. Kept for the codegen, not for a speedup.

### Motivation, from our case

`gemma-4-E2B-it` at `S=128, M=4` runs at 37 % of roofline. Fitting `time(S)` at
that configuration splits the call as **81 % fixed cost, 19 % per-key**, with a
per-key slope of ~11.2 ns that is flat from S=128 to S=32768 -- so the tile loop
is already fine and the fixed path is the whole problem.

Every shape is a compile-time `#define`, so the only genuinely runtime scalars
reaching the kernel are `S` and the block-table entries. Both carry invariants
the compiler cannot see.

### The C++ change

```c
// S is the sequence length the backend was handed, and it rejects an empty one
// before it ever reaches here.
__builtin_assume(S > 0);

// Block-table entries are page indices into the KV cache, never negative.
const int blk = __builtin_amdgcn_readfirstlane(bt[(unsigned)jb / BS]);
__builtin_assume(blk >= 0);
```

### ISA, before and after

`-DHEAD_DIM=512 -DNUM_Q_HEADS=8 -DNUM_KV_HEADS=1 -DMAXM=4 -DNSEG=4 -DMSPLIT=2`

| | instructions | branches | compares | addr math | s_waitcnt |
| --- | --- | --- | --- | --- | --- |
| before | 1077 | 19 | 27 | 38 | 54 |
| after | **1051** | **17** | 26 | 38 | 54 |

### Measured

`Hq=8 Hkv=1 D=512 M=4`, `--reps 9`, two runs each:

| | S=128 | S=1024 | S=32768 |
| --- | --- | --- | --- |
| assumes | 7.55, 7.54 | 17.24, 17.29 | 393.8, 407.4 |
| baseline | 7.58, 7.60 | 17.66, 17.76 | 409.0, 396.1 |

S=128 is -0.6 %, consistent in sign but 0.05 us -- far under the 1.66 % p90
noise floor. S=1024 is -2.5 %, the only delta that reproduces above noise.
S=32768 is indistinguishable: the runs straddle each other, and one baseline
cell reported 134 % spread, i.e. was not measurable at all.

**This is the fourth time on this kernel that instruction count has not
predicted time**, and in hindsight it could not have: the fixed path is stalled
on LGKM waits, the fence/atomic arrival protocol and a dependent global
round-trip. Removing control-flow bookkeeping from a path that is waiting on
memory changes nothing.

Kept anyway: the assumptions are true by construction, correctness is unchanged
(`max_rel` 4.80e-04 across D=64/128/256/512 at both M, mutation control fires at
3.48e+01), and strictly fewer instructions is strictly better code.

### What this rules out

Compile-time shape knowledge is not the lever at this shape. Address arithmetic
was already only 3.5 % of the kernel (38 of 1077, no sign-extensions, no integer
multiplies) -- the author had already forced 32-bit Q offsets by hand for this
reason. Together with NSEG, BLOCK, MSPLIT and fused-vs-separate reduction, the
cheap explanations for the 4.67 us of fixed cost are now exhausted; the next
step is a profiler, not another guess.

---

## 003 — Pad the LDS per-lane slice to break the bank conflict

**Status:** landed. Halves the conflict; see "reaching zero" below for the rest.

### Motivation, from our case

The profiler, not a guess. `rocprofv3` on `gemma-4-E2B-it` at `S=128, M=4`:

| | |
| --- | --- |
| kernel duration | **6.640 us** (gap to next dispatch 1.880 us) |
| warm trivial-kernel control | **1.080 us** |
| `MemUnitBusy` | 23.4 % -- not memory-bound |
| `WriteUnitStalled` | 0.0 % |
| `OccupancyPercent` | **8.62 %**, `SQ_WAVES` 256 (32 WGs x 8) |
| `LDSBankConflict` | **43.03 %** |
| `SQ_INSTS_VALU` | 1002 per wave, against 1051 static instructions |

So the time is inside the kernel, it is not memory-bound, each wave executes the
body once with no redundant work, and two things are wrong: the kernel is
work-starved at 8.6 % occupancy, and LDS is conflicting.

Occupancy is a property of the problem at S=128 -- 8 KV blocks and 8 query
heads is not enough to fill 40 CUs, and we already split 4 ways. The bank
conflict is a defect we can fix.

### The defect

`lds_acc[row * HEAD_DIM + d]` gives lane `lrow` a **blocked** slice starting at
`dl = lrow * DPL`, so the bank index `(lrow * DPL) % 32` takes only
`gcd(DPL, 32)` distinct values:

| D | DPL | write conflict | read conflict |
| --- | --- | --- | --- |
| 64 | 2 | 2-way | none |
| 128 | 8 | 4-way | none |
| 256 | 8 | 8-way | none |
| 512 | 16 | **16-way** | none |

The read (`d = tid`, consecutive) was already conflict-free, which is why
padding the *row* stride would have done nothing -- the collision is inside a
row, between lanes.

### The C++ change

```c
#define LDS_SLICE (DPL + 1)
#define LDS_STRIDE (LPR * LDS_SLICE)
#define LDS_OFF(d) (((d) / DPL) * LDS_SLICE + ((d) % DPL))

// write:  lds_acc[row * LDS_STRIDE + lrow * LDS_SLICE + i] = acc[t][i];
// read:   lds_acc[partial_row(wb, p) * LDS_STRIDE + LDS_OFF(d)]
```

`DPL` is always even, so `DPL+1` is coprime with 32 and the slice starts spread
over all 32 banks. `DPL` is a power of two, so the reader's `/` and `%` are
shifts.

### Measured

`LDSBankConflict` **43.03 % -> 20.11 %**, kernel duration **6.640 -> 5.200 us
(-21.7 %)**. Correct on D=64/128/256/512 at both M; mutation control fires.

### The -21.7 % does not reach the benchmark, and here is why

That figure is from a micro-driver that reuses one 256 KiB KV buffer, i.e.
**cache-hot**. The harness sets `min_working_set_mb=96`, which at
`S=128, D=512, Hkv=1` means `layers_for_working_set` picks **384 layers** so
every layer's KV is **cold**. Memory latency then dominates and the LDS win is
mostly hidden:

| shape | S=128 | S=16384 | S=32768 |
| --- | --- | --- | --- |
| 8/1/512 | 7.54 -> 7.29 (-3.3 %) | 200.25 -> 200.12 | 398.11 -> 388.91 (-2.3 %) |
| 16/2/512 | 8.98 -> 8.80 (-2.0 %) | 425.00 -> 417.01 (-1.9 %) | 870.38 -> 891.83 (+2.5 %) |
| 32/8/128 | 8.38 -> 8.25 (-1.6 %) | 734.50 -> 738.70 | 1466.92 -> 1460.16 |
| 8/4/256 | 6.36 -> 6.46 (+1.6 %) | 294.34 -> 295.19 | 581.47 -> 587.19 (+1.0 %) |

**~2-3 % in the regime we ship**, several cells inside the 2.9-6.6 % spread and
one slightly worse. Kept because the conflict halving is real and the code is
strictly better, not because the table is convincing.

**The methodological lesson is the bigger result: profile with the harness's
working set, or every number will be flattered.** A micro-driver on a hot cache
is measuring a different kernel than the one we ship.

### Reaching zero

Linear padding **cannot** do it, and this is provable rather than empirical.
The read needs 32 consecutive elements not to cross a shifted slice boundary,
i.e. `pad = 0 (mod 32)`; the write needs `gcd(DPL + pad, 32) = 1`, i.e.
`DPL + pad` odd. `DPL` is always even, so `pad = 0 (mod 32)` forces `DPL + pad`
even. Contradiction. Exhaustive search over `pad` in `[0, 32]` at D=512 confirms
it: every odd pad gives write 1-way / read 2-way, `pad=32` gives write 16-way /
read 1-way, and nothing gives both.

A **rotate-by-group swizzle** does, and costs no extra LDS:

```c
#define LDS_OFF(d) (((d) & ~31) | ((((d) & 31) + ((d) >> 5)) & 31))
```

Each aligned group of 32 floats is rotated by its group index. Modelled
conflict counts, write/read:

| D | DPL | LPR | pad=0 | pad=1 (landed) | rotate |
| --- | --- | --- | --- | --- | --- |
| 64 | 2 | 32 | w2/r1 | w1/r2 | **w1/r1** |
| 128 | 8 | 16 | w4/r1 | w1/r2 | **w1/r1** |
| 256 | 8 | 32 | w8/r1 | w1/r2 | **w1/r1** |
| 512 | 16 | 32 | w16/r1 | w1/r2 | **w1/r1** |

**Implemented, measured, and rejected.** The swizzle reaches exactly zero
conflicts as modelled, and is slower than the padding it would replace:

| | `LDSBankConflict` | LDS stores emitted | kernel duration |
| --- | --- | --- | --- |
| original | 43.03 % | 8 x `ds_store_b128` | 6.640 us |
| **pad=1 (landed)** | 20.11 % | 18 x `ds_store_2addr_b32` | **5.200 us** |
| rotate | **0.00 %** | 32 x `ds_store_b32` + 2 | 6.680 us |

The rotation scatters a lane's `DPL` elements across banks, which is the point,
but that also destroys the store vectorisation: 8 instructions become 34.

**That 6.680 us is cache-hot and overstates the gap.** Re-measured in the
harness, with `BFLY` retuned on each kernel and the two interleaved:

| | S=128 | S=16384 | S=32768 |
| --- | --- | --- | --- |
| pad=1 | 7.32, 7.33 | 197.9, 200.9 | 390.0, 403.7 |
| rotate | 7.49, 7.42 | 199.9, 195.3 | 395.0, 389.8 |

So rotate is ~1.5 % worse at S=128 (4 of 4 runs) and **indistinguishable
beyond**, not 28 % worse. `pad=1` still wins, but only at short context and
only slightly. The hot micro-driver exaggerated by an order of magnitude --
the same trap this entry warns about two sections above, walked into anyway.

Retuning `BFLY` on the rotate kernel was the obvious follow-up, since `BFLY=3`
had been tuned when LDS was 43 % conflicted and a free LDS pipe should make
`ds_bpermute` cheap again. It does not: on rotate, `BFLY=3` is still best
(7.45 against 7.63 for `BFLY=0`). On `pad=1` it is also still best -- `BFLY=4`
edges it at S=128 (7.28 vs 7.33) and collapses at 32k (455.9 vs 390.1). **The
LDS layout and the butterfly pipe split do not interact.** Note the padding had already
given up `ds_store_b128` for `ds_store_2addr_b32` -- it wins anyway because 18
conflict-light 2-address stores beat 8 stores that serialise 16 ways, while 34
conflict-free scalar stores do not beat either.

So the LDS pipe is priced in *instructions as well as conflicts*, and the
minimum of the product is in the middle, not at zero conflicts. `pad=1` stays.

This is the fifth measurement on this kernel where the obvious extremum lost to
a middle value -- the same shape as `BFLY`, where all-VALU (`4`) was beaten by
the three-to-one mix (`3`).

### Also tried: a pad that keeps the wide store

`pad=1` makes a slice 68 B, so it is no longer 16 B aligned and `ds_store_b128`
is lost. `pad=4` (80 B) keeps the alignment and cuts the write to 4-way, so it
should have been the best of both. Measured, it is not:

| | `LDSBankConflict` | stores | kernel | LDS/row |
| --- | --- | --- | --- | --- |
| pad=0 | 43.03 % | 8 x b128 | 6.640 us | 2048 B |
| **pad=1** | 20.11 % | 18 x 2addr_b32 | **5.200 us** | 2176 B |
| pad=4 | 20.11 % | 8 x **b128** | 5.320 us | 2560 B |
| rotate | 0.00 % | 32 x b32 | 6.680 us | 2048 B |

`pad=1` and `pad=4` tie within noise while `pad=4` costs 18 % more LDS, so
`pad=1` stays.

The identical 20.11 % is the real finding: `pad=1` writes 1-way and `pad=4`
writes 4-way, yet the counter does not move. **The residual conflict is not in
the stores.** It is the 2-way *read* conflict both share, plus `ds_bpermute`,
which runs on the same LDS hardware. That is why only the rotate reached
0.00 % -- it was the only variant that also fixed the read.

Driving the read to 1-way needs `stride = 0 (mod 32)`, the same contradiction
as above, so the rotate is the only escape and it costs more than it saves.
The two conflicts cannot be minimised together. `pad=1` is the optimum because
it buys the cheap conflict (stores) with the cheap currency (store width) and
leaves the expensive one alone.

---

## 004 — Occupancy is not the lever (refuted, do not re-try)

**Status:** rejected. `BLOCK=256 / NSEG=4` stays, which is what the heuristics
already pick.

### Why it looked promising

The profiler's headline on `gemma-4-E2B-it` at `S=128, M=4` was
`OccupancyPercent` **8.62 %** with 256 waves on a machine that holds 2560. The
obvious reading is that the kernel is starved and needs more workgroups.

Two ways to get them: raise `NSEG` (more segments, same workgroup size), or cut
`BLOCK` (same segments, more and smaller workgroups). Both were tried.

### Occupancy does rise, and the kernel gets slower

| NSEG | `OccupancyPercent` | waves/active CU | S=128 | S=32768 |
| --- | --- | --- | --- | --- |
| 4 | 9.06 % | 12.6 | **7.35** | **387.6** |
| 8 | **13.78 %** | **21.0** | 8.74 (+19 %) | 614.7 (+59 %) |
| 16 | -- | -- | 9.97 (+36 %) | 1025.9 |

Occupancy up 52 %, time up 19 % short and 59 % long. This is the cleanest
possible test of "more occupancy is better" and it fails.

### Smaller workgroups lose too

`BLOCK=128` halves `NWAVE` to 4, which also drops LDS enough that MSPLIT falls
back to 1 -- removing the forced split. `BLOCK=128 + NSEG=8` has the *same* 256
waves as the default but 64 workgroups touching all 40 CUs instead of 32:

| | S=128 | S=16384 | S=32768 |
| --- | --- | --- | --- |
| BLOCK=256 NSEG=4 | **7.34** | **199.0** | **397.0** |
| BLOCK=128 NSEG=4 | 9.04 (+23 %) | 271.8 (+37 %) | 530.4 (+34 %) |
| BLOCK=128 NSEG=8 | 10.77 (+47 %) | 289.6 | 588.1 |

The equal-wave, better-spread cell is the **worst** of the four.

### Larger workgroups trade, they do not win

| | S=128 | S=16384 | S=32768 | spread |
| --- | --- | --- | --- | --- |
| NSEG=4 BLOCK=256 | 7.34 | **192.8** | **393.2** | 4.4 % |
| NSEG=4 BLOCK=512 | **6.54** (-10.9 %) | 215.9 (+12 %) | 441.4 (+12 %) | 4.8 % |
| NSEG=2 BLOCK=512 | 6.41 | 295.0 | 585.1 | **1117 %** |

`BLOCK=512` is really better at short context and really worse at long. One
value must serve S=128 through S=32768, and scoring the worst context keeps
256. `BLOCK=512` is also unstable -- 1117 % spread on one cell, 11.8 % on
another. `BLOCK=1024` does not build: `MAXM=4` admits only MSPLIT in {1,2,4}
and none fits LDS at `NWAVE=32`.

### Why

Every workgroup pays a prologue, an LDS reduction and a partial publication
whose cost is independent of how much KV it covers. Splitting further divides
the work but not the overhead, and at long context it fragments the KV locality
the design rests on. The intra-workgroup LDS reduction is cheaper than the
cross-workgroup one, so pushing work *out* of the workgroup is the wrong
direction -- which is why `NSEG` up and `BLOCK` down fail for the same reason.

**8.62 % occupancy is the correct operating point for a problem with 8 KV
blocks and 8 query heads.** It is a symptom of the problem being small, not a
cause of the kernel being slow. The grid space is now exhaustively swept:
`BLOCK` in {128, 256, 512} x `NSEG` in {1, 2, 4, 8, 16}.

### Harness bug this uncovered

The override path did `replace(self._variant, **current)` on a variant that had
already been through `__post_init__`, which raises MSPLIT until the partials fit
LDS. That raised value was carried forward, so overriding `BLOCK` kept the
MSPLIT that `BLOCK=256` needed instead of re-deriving the one `BLOCK=128`
allows -- the runtime asked for `_b128_ms2_` when the correct variant is `ms1`.
`precompile` starts from the heuristic knobs and the override did not, so they
disagreed. `sweep.py` now rebuilds from `_knobs_for` before applying the
override. Caught by the seal; without it this would have silently measured the
wrong variant.

---

## 005 — Dispatch the grid head-fastest for one configuration

**Status:** landed, as a `_TUNED` row for `(8, 2, 512, 1)` only. Neutral across
the matrix; it is not a rule.

### Why

`gemma-4-E4B-it` at M=1 was the lone outlier of the D=512 table: **53.2 % of
roofline and 1.60x** against Triton at S=32768, where every neighbour sat at
90-99 % and ~3x. Nothing in the shape explains it — `16/2`, same Hkv and same
D, was at 99 %.

The grid is `(NSEG, NUM_Q_HEADS)` and HIP dispatches x fastest, so consecutive
workgroups differ in *segment*. Every q head of one kv head reads
byte-identical addresses, while two segments of one head read merely adjacent
ones (ILV interleaves them every `KPWE*SUB` tokens). Transposing the grid makes
the identical readers consecutive instead of the adjacent ones.

### The C++

```c
#if GRIDT
  const int seg = blockIdx.y;
  const int h = blockIdx.x;
  dim3 grid(NUM_Q_HEADS, NSEG), block(BLOCK);
#else
  const int seg = blockIdx.x;
  const int h = blockIdx.y;
  dim3 grid(NSEG, NUM_Q_HEADS), block(BLOCK);
#endif
```

### ISA before and after

Identical: **721 instructions, 102 VGPR** either way. The only difference is
which SGPR carries which block index — `s2` and `s3` swap roles:

```asm
- s_load_b64  s[22:23], s[0:1], 0x40      + s_load_b64  s[6:7], s[0:1], 0x40
- s_mov_b32   s20, s3                     + s_mov_b32   s4, s3
- v_lshl_or_b32 v1, s2, 3, v74            + v_lshl_or_b32 v1, s4, 3, v74
```

This is the rare knob that is free in code and large in time, which is exactly
why it had to be measured rather than reasoned about.

### What it measured

`Hq=8 Hkv=2 D=512 M=1`, all seven contexts, against the previous golden:

| S | 128 | 512 | 1024 | 4096 | 8192 | 16384 | 32768 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| change | -1.0 % | -1.7 % | -1.3 % | -5.2 % | -8.6 % | -21.7 % | **-45.3 %** |

Monotonic in context, regresses nothing, and confirmed in a second independent
run (1042.01 -> 564.18 us at S=32768). The cell now reads 95.8 % of roofline
and 2.80x. D=512 matrix: M=1 geomean **2.888x -> 2.963x**, still 0 losses in 35
cells; M=4 unchanged within noise.

### The mechanism, profiled

`FETCH_SIZE` at S=32768, distinct KV 128 MiB:

| | order | reuse | us | GB/s |
| --- | --- | --- | --- | --- |
| 8/2 M=1 | segment-fastest | 2.78x | 1024.7 | 364 |
| 8/2 M=1 | head-fastest | **1.00x** | 552.9 | 243 |
| 16/2 M=1 | segment-fastest | **1.00x** | 543.6 | 247 |
| 16/2 M=1 | head-fastest | 1.43x | 616.7 | 311 |

Both configurations *can* stream perfectly at ~245 GB/s, the bus limit; each
needs the opposite order. The knob does not create value, it decides which
configuration is well matched — which is why the matrix geomean is 1.001.

### Why it is a table row and not a rule

Forced on every configuration it costs 7.0 % on `(8,1,512,4)`, 3.9 % on
`(16,2,512,1)` and 4.3 % on `(16,2,512,4)`, and is within +-1 % on the other
six pairs — including `(16,1,512,*)`, the cell this investigation started from.

Nothing separates the winner from the losers. `GQA`, `NSEG` and workgroups per
kv head each take the same value in at least one winner and one loser. The only
pattern is that the winner is the sole configuration with `GQA == NSEG`, which
is **n = 1** and exactly the shape of reasoning that put the WGP-alignment rule
in the design document three refutations ago. Do not promote it to `_knobs_for`
without more points.

### Not measured

D=64, D=128 and D=256. The exploration ran them with `GRIDT=1` forced but no
baseline was taken to compare against.

---

## 006 — Widen the KV tile to eight keys per wave (rejected)

**Status:** rejected. `KPW=4` stays everywhere; `kpw` is deliberately *not*
added to `_Knobs`.

### Why it looked promising

At `16/1/512` and M=4, the configuration this investigation started from, a
focused sweep over 2048/16384/32768 with `--reps 3` put `KPW=8` ahead at every
point: -3.3 %, -2.4 %, -8.4 %. A wider tile amortises the block-table read and
the address arithmetic over twice the keys and keeps twice the loads in flight,
which is the one thing the long-context end of this kernel is short of.

### ISA

| KPW | instructions | `global_load_b128` | VGPR | spill |
| --- | --- | --- | --- | --- |
| 4 | 1340 | 22 | 127 | 0 |
| 8 | 1823 | 38 | 192 | 0 |
| 16 | 2972 | 70 | 256 | **93 B** |

16 spills and is 3.5x slower; it is not a candidate. 8 is clean.

### What killed it

Forced across the whole D=512 matrix, `KPW=8` is not a global win at all:

| | geomean vs Triton | losses |
| --- | --- | --- |
| M=1, KPW=4 | **2.963x** | 0 |
| M=1, KPW=8 | 2.629x | 1 |
| M=4, KPW=4 | **2.928x** | 0 |
| M=4, KPW=8 | 2.781x | 0 |

Eight of the ten configuration/M pairs regress, several severely (`8/2` at M=4
and `16/2` at M=1 both peak above +113 %).

Two pairs looked landable and neither survives the rule in HANDOFF §4.1 --
choose the value that regresses no cell by more than the ~3.3 % harness noise:

- `32/4` M=4: -6 % to -14 % from S=128 to S=8192, then **+5.0 % at 16384 and
  +39.0 % at 32768**. A large, unambiguous long-context regression.
- `16/1` M=4: monotonic in context, -10.7 % at S=32768, but **+4.6 % at
  S=128**. Re-measured on its own with `--reps 5`: 8.32 -> 8.70 us, the same
  +4.6 %. Reproducible, so it is a real trade, not noise.

The second is the interesting one and it is still a reject. Trading 4.6 % at
short context for 10.7 % at long is exactly the "knowingly bad trade at some
context" the rule exists to refuse, and the one-configuration-per-shape
constraint means it cannot be taken only where it pays.

There is no middle value to fall back on: `BS % KPW == 0` with `BS=16` admits
only 4, 8 and 16.

### Tooling

`matrix.py` grew `--kpw` alongside `--gridt`, both of which override
`_knobs_for` for the whole run. Measuring a knob against the golden across all
70 cells is what separated this from the three-context sweep that made it look
like a win.

---

## 007 — Shorten the score butterfly by widening the lane slice

**Status:** landed on four of the five D=512 M=4 configurations, as
`{"dpl": 32, "ldsplit": 2}` rows. M=1 keeps `DPL=16`.

### How the ablation found it

`ABLATE` thins one VALU block to a single iteration and leaves the loads, the
loop and the epilogue intact. It returns wrong numbers on purpose — every bit
must make `check.py` fail, and all three do — so the delta bounds what
restructuring that block could ever buy. `16/1/512` at M=4:

| ablated | S=8192 | S=32768 | |
| --- | --- | --- | --- |
| nothing | 192.33 | 771.35 | |
| P@V (16 -> 1) | 215.71 | 837.84 | **+12 % slower** |
| Q@K (8 -> 1) | 212.66 | 821.77 | **+11 % slower** |
| butterfly (5 -> 1 stages) | 163.29 | 684.24 | -15 % / -11 % |

**Removing arithmetic makes the kernel slower in two of three cases.** That
work is hiding memory latency for free; take it away and the wave stalls on the
loads instead. It is the sharpest evidence yet that this kernel is not
VALU-throughput bound, and it refuted a model — `time ~ max(bytes/BW, M*c)` —
that had fitted M=1, M=2 and M=4 to within 10 % an hour earlier. The fit was a
coincidence.

Only the butterfly costs real time, and the reason is structural: its strides
are a dependency chain (stride 2 consumes stride 1), so unlike P@V and Q@K it
cannot be overlapped with anything.

### The change

`DPL` is fp16 per lane of a row; `LPR = HEAD_DIM/DPL` lanes cover one row and
the butterfly runs `log2(LPR)` stages. At D=512 the rule gave `DPL=16`,
`LPR=32`, five stages. `DPL=32` gives `LPR=16` and **four**.

```c
#ifndef DPL
  #define DPL (HEAD_DIM == 128 ? 8 : HEAD_DIM / WAVE)
#endif
```

`SUB = WAVE/LPR` doubles to 2, so the partial count doubles and LDS overflows:
32 rows x 528 floats + scalars = 67840 B against a 65536 B ceiling. `LDSPLIT=2`
chunks the epilogue's reduction over the head dimension and brings it to
34052 B. LDSPLIT is an enabler, not an optimisation — on its own it measures
neutral (191.55 against 193.97 us).

| | instructions | LDS | VGPR | spill |
| --- | --- | --- | --- | --- |
| DPL=16 | 1329 | 34948 | 127 | 0 |
| DPL=32 + LDSPLIT=2 | 1760 | 34052 | 193 | 0 |

### What it measured

D=512 matrix, against the previous golden:

| | geomean vs Triton | worst | losses |
| --- | --- | --- | --- |
| M=1 before / after | 2.963x / **2.965x** | 1.65x | 0 |
| M=4 before / after | 2.928x / **3.044x** | 1.91x -> **2.13x** | 0 |

Per configuration at M=4, and why one is excluded:

| Hq/Hkv | geomean | worst cell | landed |
| --- | --- | --- | --- |
| 16/1 | 0.915 | +0.54 us | yes |
| 32/4 | 0.944 | +0.47 us | yes |
| 16/2 | 0.970 | +0.69 us | yes |
| 8/1 | 0.989 | +0.66 us | yes |
| 8/2 | 1.015 | +6.86 us | **no** — regresses five of seven contexts |

`16/1` at M=4, the configuration this whole investigation started from, goes
from 33.6 % to 40.8 % of roofline at S=32768 and 815.7 -> 670.6 us.

### Why M=4 only

At M=1 the kernel already streams at 94 % of the bus, so a shorter butterfly
buys nothing while 193 VGPRs against 127 and the chunked epilogue cost real
time. It loses on all five M=1 configurations, 2.5-6.6 %.

### It changed the acceptance rule

These rows regress S=128 by 4-9 %, which the old percentage rule forbade. In
absolute time that is 0.47-0.69 us against 137-203 us saved at S=32768. HANDOFF
§4.1 now reads in microseconds; see the note there.

`KPW=8` (entry 006) was re-examined under the new criterion and stays rejected
on its own merits: it does not combine with `DPL=32` — `SUB` doubles so `KPWE`
does too — and the combination regresses up to +676 us, while on `16/1` alone
`DPL=32` is simply better (0.915 against 0.981).

### Not done

D=64, D=128 and D=256 were not measured at all. `BFLY` was re-tuned under the
shorter butterfly in entry 008.

---

## 008 — Re-tune BFLY under the four-stage butterfly

**Status:** landed. `(8,1,512,4)` and `(16,2,512,4)` go from `BFLY=3` to
`BFLY=0`. `(16,1,512,4)` and `(32,4,512,4)` keep 4.

### Why re-sweep at all

Entry 007 halved `LPR`, so the butterfly is four dependent stages instead of
five and each stage carries a different balance of work. `BFLY` splits that
work between the two cross-lane pipes -- `ds_bpermute` on the LDS pipe against
`v_permlane16` on the VALU -- so changing the number of stages changes what the
right split is. Entry 003 set the precedent: an LDS change invalidated one
neighbouring `BFLY` row and the rest had to be re-checked.

No code changed here. This is the knob being re-measured in its new regime.

### What it measured

`BFLY` 0..4, four configurations, seven contexts, `--reps 1`, against the
golden:

| Hq/Hkv | current | best | geomean | worst cell |
| --- | --- | --- | --- | --- |
| 8/1 | 3 | **0** | 0.9652 | -0.07 us (nothing regresses) |
| 16/2 | 3 | **0** | 0.9657 | -0.17 us (nothing regresses) |
| 16/1 | 4 | 4 | 0.9990 at bfly=0 | +1.59 us |
| 32/4 | 4 | 4 | 1.0289 at next best | -- |

Both moves confirmed in a second independent run: `8/1` wins -1.1 % to -5.8 %
across all seven contexts, `16/2` -1.9 % to -8.6 %. D=512 matrix afterwards:
M=4 geomean **3.044x -> 3.081x**, 0 losses in 35 cells; M=1 unchanged.

`BFLY=0` puts the whole butterfly on the LDS pipe, which the main loop still
never touches at any `BFLY`. A four-stage reduction leaves the VALU with more
register pressure per element than a five-stage one did, so the idle pipe is
worth more than it was -- the same argument entry 001 used, landing on the
other extreme now that the shape changed.

### The reading this corrected

`16/1` had looked 6.7 % better at `bfly=0` in a three-point sweep whose cells
carried 10-13 % spread, and that reading went into the golden's caveats as a
follow-up worth taking. A clean seven-context pass puts it at 0.9990. It was
noise, and a knob with 10 % spread on the deciding cell is not a finding.

### Cost

269 s for the four-configuration sweep, 45 s for the confirmation once the
builds were cached. An earlier attempt at `--reps 5` was abandoned: reps
resample allocation and graph placement, not `do_bench`'s within-cell
dispersion, so they cost 5x and do not resolve what a second independent run
resolves for 45 s. See HANDOFF §2.

---

## 009 — Rewrite: one workgroup per kv head, WMMA for both products

**Status:** landed. Replaces the kernel entries 001-008 describe; their knobs
(`BFLY`, `GRIDT`, `DPL`, `LDSPLIT`, `KPW`, `MSPLIT`, `ILV`) no longer exist.
Those entries stay as the record of what that kernel learned.

### Motivation, from our case

The old grid was `(NSEG, Hq)`: one workgroup per *q* head, each streaming its
kv head's KV and relying on L2 to absorb the GQA-fold re-read. Its `%roof` fell
as `1/GQA` (HANDOFF §6, golden d512) and the matrix before this entry had
**2 of 52** rows at a 90 % geomean of roof; `32/2/128` sat at 28.9 % (M=1) and
23.8 % (M=4), `16/1/512` at 40.5 % (M=4). At M=4 the per-q-head VALU work --
Q@K, a five-stage score butterfly and P@V per row -- was the limit, not bytes.

### What it does

One workgroup per `(kv head, row group, KV segment)`. All `GQA x M` rows that
read a kv head go through the same workgroup, so the KV is read from memory
once. Both products are `v_wmma_f32_16x16x16_f16`:

    S^T[key][row] = K[key][:] . Q[row][:]      A = K tile, B = Q^T
    O^T[d][row]  += V[key][d] . P[row][key]    A = V^T,    B = P^T

Each wave owns 16-key tiles and runs its own online softmax, so the loop has
no barrier; the waves merge once at the end. `DSPL` waves may share a tile,
each owning `D/DSPL` of the head dim (needed at D>=256 to keep a wave's
accumulator and K/V slice in registers); they sum their partial scores
through LDS. `NSEG` is a maximum: the kernel activates
`clamp(nblocks/MINB, 1, NSEG)` segments from S at run time, so short contexts
skip the cross-workgroup merge that long ones need, with a grid fixed at
graph capture.

### What each piece is worth, measured

`32/8/128` M=1 unless stated, dev harness (same method as matrix.py: HIP
graph, >=96 MiB rotated working set, arange block table), geomean %roof over
the seven contexts.

| step | geomean | note |
| --- | --- | --- |
| old kernel | 69.5 % | matrix before this entry |
| first cut: V^T gathered with b16 loads | 53.6 % | 128 VMEM instructions per wave per block |
| each wave owns its tiles, V^T built with v_perm from b128 rows | 36.2 % | LDS float atomics in the merge, see below |
| merge by tree instead of `ds_add_f32` | 63.7 % | |
| loads forced ahead of use (`asm volatile("" ::: "memory")`) | 76.7 % | the scheduler had issued K two loads at a time |
| merge stores real rows only, one round | 80.0 % | LDS 56 KB -> 18 KB |
| `NSEG` for 16 workgroups, `MINB=4` | 80.8 % | |
| K read row-wise, transposed through LDS | 82.7 % | 96.7 % of roof at S=32768 |
| V then K, not interleaved | 83.8 % | |
| `NW=4`, `RG=2` | 85.6 % | |

`32/32/128` M=1 reaches **93.2 %** at `NW=2`, the first configuration over 90.

### Findings worth keeping

**`ds_add_f32` costs ~17 us per call.** The first merge had every wave add its
scaled accumulator into LDS with float atomics, 64 per lane. Replacing them
with plain stores (wrong answers, timing only) took S=128 from 24.0 us to
7.0 us. Removing the 16-way bank conflict first changed nothing, so it is the
atomic path itself. Never use LDS float atomics in this kernel.

**The compiler serialises loads under VGPR pressure.** At 250 VGPRs it issued
a tile's K as two `global_load_b128`, `s_waitcnt vmcnt(0)`, one WMMA, repeat:
eight round trips per tile, and V only after Q@K. An empty `asm volatile`
with a memory clobber after the loads pins them: 53.6 -> 76.7 %.

**K lane-per-key costs ~5 % of DRAM efficiency.** The WMMA A operand wants
lane = key, 16 consecutive d in-lane, so a direct load touches 16 rows per
instruction. Loads only (`ABLATE=64`), `NSEG=2`, S=32768: 92.8 % of roof that
way, **97.0 %** reading K row-wise like V. The kernel now reads K row-wise and
transposes it through a 4.3 KB per-wave LDS tile (no barrier: one wave's LDS
ops complete in order).

**Fewer, longer streams.** A tile-structured stream with this kernel's access
order: 97.9 % of peak at 16 workgroups, 96.4 % at 40, 90-94 % at 64 and up.
Hence `_TARGET_WORKGROUPS = 16`.

**Issue order within a tile matters.** All of V then all of K, or the
reverse: S=128 on `32/32/128` at 70.7 %. Interleaved row by row, same bytes,
same addresses: 60.3 % -- the last KV byte landed 1.7-3.3 us later.

**Each half of a wave reads its own copy of the WMMA operands.** Measured
with garbage in chosen lanes: the lower half computes the even output rows
from its own A (only the even rows of it) and its own B, the upper half the
odd rows from its own. So the two halves may order the k index differently
as long as each half's A and B agree. The kernel uses "this half's keys
first": every operand becomes (own, other half's) in every lane with no select
on the half. `v_cndmask` per tile 122 -> 48; interleaved A/B on `32/32/128`
92.7/92.3 % against 90.7/91.5 %.

**The split-KV merge in one L2 round trip.** The last segment to arrive
used to read a running max, then weights, then partials: three dependent L2
round trips.  Issuing every segment's m, l and partial at once (NSEG
unrolled): `8/1/256` M=1 55.6 -> 63.7 %, `32/4/128` 76.3 -> 78.7 %.

**P needs more than fp16.** One fp16 P misses the 1e-3 relative bound where
the output is near zero (7.9e-2 at S=5). P goes as fp16 high plus fp16 low
half, two WMMAs -- or one, when a row tile has at most 8 real rows: the low
half rides in the padding columns and is folded back once at the end
(`PPACK`). The error is then 4.8e-4, the fp16 output rounding floor, as
before.

**Row groups and d splits for large GQA x M.** An accumulator of 16 rows at
128 d is 64 VGPRs; 64 rows spilled (`32/2/128` M=4: 18.1 %). `RG` splits rows
over workgroups that share the KV in L2 (57.5 % at `RG=4`); where GQA has no
fitting divisor (7, 5) `DSPL=2` halves each wave's d instead (28/4 M=4: 63.0 %
against 47.9 % at `RG=7`).

### Rejected

| idea | result |
| --- | --- |
| two tiles in flight per wave (double buffer) | 70.7 % against 80.0 % at S=128 on `32/32/128`; 256 VGPRs and spills at D=128 |
| next tile's K and V held until the current P@V is done | 59.2 % against 77.4 % (`32/4/128`); still spills |
| next tile's K issued right after Q@K | neutral |
| both halves load all 16 V rows (no exchange) | 92.0 % against 93.2 %, and 79.3 % against 85.6 % |
| Q loaded before the page table | neutral |
| split-KV merge with weights precomputed once | neutral, kept for the independent loads |
| `NW=16` | does not fit: the per-wave K tiles alone are 69 KB |
| count arrivals first, fence only the non-last segments | 78.2 against 78.6 %, 62.1 against 63.7 %: their fence lands on the last one's path |
| `RG = GQA` (one q head per workgroup, the old decomposition) | 29-36 % at D=256/512 M=1 |
| skip the DSPL score exchange (wrong answers, bound) | no gain: the barriers are not the cost |

### Against the old kernel

D=512 from `matrix.py`, against `reference/golden_d512_dot.md`, geomean of
roof: M=4 59.7 -> 71.1 % (`16/1` 1.53x faster, `32/4` 1.43x), M=1
82.9 -> 73.3 %.  At M=1 it loses on four of five configurations, 0.76-0.93x,
nearly all at S=128-1024, where few real rows (GQA x M <= 16) leave most of
each WMMA as padding and its latency, and the merges, sit on the critical
path.  From S=8192 the two are level.

### Traps met on the way

- A `?:` between a lane's own value and a `permlane16` of it compiled to a
  branch: the value was computed only in the lanes that took it and read
  from the lanes that did not. Select with a mask.
- The K tile is stored as integers and read as halves; type-based alias
  analysis let the reads move above the stores. An `asm` memory clobber
  between them.
- `amd-gpu-lock` is not a mutex: it polls for other GPU processes, so two
  jobs polling together both start. Measurements taken while another job ran
  scattered by +-2 % at long context; serialised with `flock` they repeat to
  0.1 %.

---

## 010 — Trim the preamble and the softmax

**Status:** landed, `e79bc3cb18`.

### Motivation, from our case

At S=128 the effective work is one or two 16-key tiles per wave; the rest of
the ~3.4 us inside the kernel is preamble, waits and epilogue. The ISA showed
four costs that did no work.

### What changed, and what each measured

Interleaved A/B against the previous kernel, dev harness (HND), S=128 unless
stated.

**Barriers order LDS only.** `__syncthreads()` compiles to
`s_waitcnt vmcnt(0) lgkmcnt(0)`, `s_barrier`, `buffer_gl0_inv`: it waits for
every outstanding global load and invalidates L0. Every barrier in this kernel
orders LDS only (global visibility for the merge is `__threadfence`'s job),
and the one that publishes Q therefore waited for the whole first KV tile to
land before any wave could start Q@K. All eleven became

    s_waitcnt lgkmcnt(0)
    s_barrier

**Q loaded unconditionally, from a clamped row.** Under `if (r < ROWS_W)` the
compiler waited `vmcnt(0)` -- Q and the page table -- before it issued the
first KV load; unconditional, it waits `vmcnt(1)` for the page table only.
Padding rows are zeroed after the load.

Together: `32/8/128` M=1 +9.9 %, M=4 +5.5 %, `16/1/512` M=4 +4.9 %, the other
case studies +0.8 to +1.4 %. Correctness unchanged (max_rel 4.9e-4).

**permlane with fetch-inactive.** `v_permlane16_b32` / `v_permlanex16_b32`
with `fi=0` must preserve the destination in inactive lanes, so the compiler
tied the destination to a copy of the source: one `v_mov` per permute. Every
source lane is active in every use, so `fi=1` (`op_sel:[1,0]` in the ISA)
changes nothing but the copies: `v_mov` 207 -> 141 on `32/8/128`, VGPRs
unchanged at 237. `32/8/128` +1.0 %, the rest within noise.

**Causal mask on the tail tile only; the scale inside the exponent.** Only a
tile that reaches past the first query token's keys can hold a masked key,
so the per-element compare and select moved under a uniform branch. The
scale multiplies inside the exponent as `fma(s, scale2, -m)` (it is
positive, so the max commutes): one FMA replaces a multiply and a subtract.
+0.3 to +0.8 %.

### Rejected in the same pass

| idea | result |
| --- | --- |
| lazy rescale: move the running max only when a row grows by > 2^8 (skips 64 `v_mul` per tile) | neutral at S=128, -1 to +1 % at S=4096-16384 on M=4: the loop is not VALU-bound |
| one select for rows with every key masked (`mref`), instead of one per element | neutral to -2.7 % at S=8192 |

---

## 011 — Share the split-KV merge across the segments

**Status:** landed, `c17a8bda13`.

### Motivation, from our case

`TIMING=3` on `16/1/512` M=4, S=128 (NSEG=8, RG=4, NW=8): the loop ended at
~2.8 us, and the kernel at 6.8 us. The last segment to arrive then read every
segment's partial alone -- 16 rows x 512 d x 8 segments x 4 B = 256 KiB into
one workgroup, ~2 us, bound by one CU's bandwidth. Unrolling the merge loop
so more loads are in flight changed nothing (0.97-1.03x), which is what a
bandwidth limit looks like.

### What it does

When a group's partials reach 64 KiB (`SHARED_MERGE`, compile time), the
segments wait for each other and each merges its own slice of rows x D:

- the last to arrive bumps a per-group **generation** word, then resets the
  arrival counter (the order matters: a release store behind the counter
  reset would wait for its ack);
- the others spin on the generation (`s_sleep 1` between polls);
- every segment then merges `ROWS_W x D / nseg` elements.

The generation is read at kernel start: it cannot move before this segment
arrives, so the read costs nothing on the way to the arrival (+2 to +3 % at
S=128 against reading it there). The counter buffer doubles to hold the
generations; they are compared for change, never reset.

Waiting is only safe if every workgroup of the grid is resident at once. The
host checks it (`coop`) and falls back to the last-arriver merge otherwise.
HIP's `hipOccupancyMaxActiveBlocksPerMultiprocessor` reports 1 workgroup per
WGP for anything over 32 KiB of LDS -- it assumes 64 KiB per WGP, gfx1151 has
128 -- so the host also counts `floor(128 KiB / LDS)` and the wave limit from
the kernel's VGPRs (1536 per SIMD in blocks of 24), rounding every way down,
and takes the larger. `16/1/512` M=4 (52.6 KB LDS, 206 VGPRs): API 1,
counted 2, grid 32 against 40.

### Measured

Interleaved A/B against 010, dev harness:

| configuration | S=128 | S=1024 | S=8192 |
| --- | --- | --- | --- |
| `16/1/512` M=1 | **1.49x** | 1.31x | 1.06x |
| `8/1/512` M=1 | 1.20x | 1.13x | 1.03x |
| `8/1/512` M=4 | 1.16x | 1.14x | 1.03x |
| `16/1/512` M=4 | 1.15x | 1.11x | |
| `8/2/512` M=4 | 1.14x | 1.10x | 1.01x |
| `8/2/512` M=1 | 1.11x | 1.04x | 1.01x |
| `32/4/512` M=1 | 1.09x | 1.02x | 1.00x |
| `16/2/512` M=4 | 1.09x | 1.04x | 1.01x |
| `16/2/512` M=1 | 1.06x | 1.03x | |

After it, the same timeline ends at 5.6 us: the merge takes ~0.7 us, and the
fence, atomic and wait before it ~1-1.3 us.

### Why a threshold

Below ~64 KiB the lone merger is faster: the wait costs a round trip the
merge no longer repays. With the shared merge forced on, `16/2/64` M=1 (16
KiB of partials) measured -4 %, `8/1/256` M=1 (32 KiB) -2 to -3 %,
`16/8/256` M=1 (4 KiB) -1 %. At 64 KiB (`16/2/512` M=1, `32/4/512` M=1)
it already wins.

### Rejected

| idea | result |
| --- | --- |
| unroll the merge loop 2 or 4 times | 0.97-1.03x: bandwidth-bound, not latency-bound |
| order the grid segment-fastest, so a group's segments dispatch together (would make the wait safe for any grid under in-order dispatch) | -2 to -3 % on `16/2/512` M=1 at S=128; rg-fastest-then-segment likewise |

### ISA

The wait, as the last-arriving workgroup's peers run it:

    .LBB0_41:
        s_sleep 1
        global_load_b32 v3, v2, s[16:17] offset:4 glc
        s_waitcnt vmcnt(0)
        v_cmp_eq_u32_e32 vcc_lo, v3, v1
        s_cbranch_vccnz .LBB0_41

### Tests

`test_shared_split_merge` (16/2/512, M=4, nseg=8, three launches) reaches the
shared path: shifting each segment's slice by four elements on purpose makes
it fail at S=48 and S=1024.

---

## 012 — `16/2/512` M=1 to one row group, and why RG > 1 is fragile

**Status:** landed, `16621e4856` (knobs only).

### Motivation, from our case

After 010, `16/2/512` M=1 lost at long context: 0.93x at S=8192, 0.87x at
16384, **0.78x at 32768** against the previous kernel. It was the only
configuration measured that did.

### What it was

Bisected by barrier: any one `__syncthreads()` in place of `lds_barrier()`
before or inside the main loop restores it; one in the epilogue does not;
`vmcnt(0)` or `buffer_gl0_inv` written in `asm` at the same place does not
either. So it was not what the barrier waits for, but how its presence
changed the rest of the kernel's code -- and through it, timing.

The counters showed what timing changed. `16/2/512` M=1 at S=32768:

| | `GL2C_EA_RDREQ_DRAM` | `GL2C_HIT` | `GL2C_MISS` |
| --- | --- | --- | --- |
| LDS-only barriers | 1.68 M | 0.44 M | 1.68 M |
| `__syncthreads` at the Q barrier | 1.05 M | 1.07 M | 1.05 M |

1.05 M reads of 128 B is exactly the 128 MiB of KV. With `RG=2` the two row
groups of a kv head read the same KV and rely on L2 to fetch it once; with
LDS-only barriers they drifted apart and read it 1.6 times. Six other RG>1
configurations stayed at 1.00-1.04x either way. Whether the row groups share
depends on how their workgroups drift, which nothing in the kernel controls.

### What changed

`RG=1`, which does not depend on the sharing: `nseg 8, rg 1, minb 1, nw 4`.
`matrix.py`, HND, against the golden run before 010: S=128 48.7 -> 55.2 %roof,
S=32768 94.2 -> 96.3 %, better at every context.

### Rejected

| idea | result |
| --- | --- |
| `__syncthreads` back at the Q barrier, for every configuration | fixes `16/2/512` M=1 (1.29x at S=32768) and costs 3-8 % at S=128 elsewhere (`32/8/128` -8 %, `32/2/128` M=4 -6 %) |
| force the page-table load to VMEM (a uniform address had become `s_load`, counted in `lgkmcnt` with LDS, so the loop's barriers waited for it) | removes that wait, not the regression |
| a `__builtin_amdgcn_s_waitcnt(vmcnt(0))` before the loop (the waitcnt pass merged the preamble's load order into the loop header and waited `vmcnt(0)` every iteration; it cannot see a wait in `asm`) | loop waits identical to the good build, regression unchanged |
| issue K before V | neutral |

### Open

Under the NHD layout (vLLM's default), with shuffled pages, 010-011 are 4-7 %
slower than before at S >= 16384 on `8/4/256` and `16/8/256` M=4; knobs do not
recover them. HND, which every number here uses, does not show it. Not
bisected.

---

## 013 — bf16

**Status:** landed, `88fb3dc37b`.

### Motivation, from our case

The kernel took fp16 only: the host op refused bf16 and the backend sent every
bf16 model to Triton. Most of `tools/shapes.csv` ships in bf16, so for them
none of 009-012 applied.

### What changed

The element type is a compile-time define, `KV_BF16`, like every other shape
parameter: one build serves one dtype, `KernelVariant.dtype` names it (`_bf16`
suffix) and the backend picks it from the query, refusing a KV cache of
another type. fp16 builds compile to the same ISA as before, byte for byte, on
six variants covering D=64 to 512, PPACK and DSPL.

What moves is bits, so loads, LDS staging and every permlane stay as they
were. Three places depend on the type:

- **The products.** `wmma()` wraps `v_wmma_f32_16x16x16_bf16` (operands as
  16 x `short`, the builtin's canonical type) or the f16 one.
- **P.** P keeps its high + low split (PPACK or the second WMMA): ~16 bits in
  bf16 against 8 for one bf16 P. gfx1151 has no f32 -> bf16 conversion, so the
  high half **truncates**: two values pack in one `v_perm_b32`, and the low
  half, `p - (p & 0xffff0000)`, is exact in fp32 and carries what truncation
  drops. Rounding would cost ~5 VALU per value in the loop; the truncated
  pair is

      v_and_b32_e32 v175, 0xffff0000, v76          ; high half, as a float
      v_perm_b32    v188, v89, v81, 0x7060302      ; two of them, packed

- **The output.** Round to nearest even. The compiler's `(__bf16)` cast also
  keeps a NaN a NaN, a compare and a select more per value:

      ; (__bf16)x                          ; to_elem(x)
      v_bfe_u32   v2, v0, 16, 1            v_bfe_u32  v10, v1, 16, 1
      v_or_b32    v9, 0x400000, v0         v_add3_u32 v0, v1, v10, 0x7fff
      v_cmp_u_f32 vcc_lo, v0, v0
      v_add3_u32  v2, v2, v0, 0x7fff
      v_cndmask_b32 ...

  `to_elem` drops the NaN branch; a NaN may come out as Inf, still not
  finite, which is all the tests ask of it. Epilogue only: 1083 against 1112
  instructions on `32/8/128` M=1, 1851 against 2048 on `16/2/512` M=1.

### Precision

The reference sees the same rounded inputs, so what differs is the kernel's
arithmetic and the output rounding, which alone is up to 2^-8 = 3.9e-3
relative. The bound is 8e-3 for bf16 (1e-3 stays for fp16).

`check.py --dtype bf16`, all 52 `_TUNED` configuration/M pairs, S = 48, 1000
(partial tile, NaN past the sequence) and 4096, both layouts, two launches:
worst max_rel **4.4e-3**; most cells sit at 3.8-3.9e-3, the output rounding.
S=5 reaches 5.3e-3 on `16/1/512` M=4. The `--mutate 1` control at M=4 is
detected on all 26 configurations, the smallest max_rel 23.

### Measured

Interleaved A/B against the fp16 build of the same configuration (same knobs,
HND, shuffled pages, KV rotated over >= 96 MiB, `do_bench_cudagraph`, five
rounds alternating), all 52 pairs. bf16 speed-up over fp16, median (worst):

| output conversion | S=128 | S=1024 | S=8192 | S=32768 |
| --- | --- | --- | --- | --- |
| `(__bf16)` cast | 0.995 (0.989) | 0.999 (0.993) | 1.000 (0.982) | 1.001 (0.996) |
| `to_elem` | 0.997 (0.989) | 0.999 (0.986) | | 1.000 (0.992) |

`to_elem` against the cast, same configuration and context: +0.25 % median at
S=128, better on 39 of 52; neutral at long context. What is left at S=128 is
the conversion itself; the loop costs nothing.

`matrix.py --dtype bf16`, HND, all 52 pairs with Triton in bf16 too
(`golden/bf16.md`); the fp16 column is `golden/d*.md`, a different run:

| D | M | configs | vs Triton | median configuration %roof | fp16 (`golden/d*.md`) | >= 90 % roof | S=128 median %roof |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 1 | 4 | 1.27x | 84.0 % | 83.7 % | 1 | 55.5 % |
| 64 | 4 | 4 | 1.29x | 83.5 % | 83.8 % | 1 | 56.6 % |
| 128 | 1 | 10 | 1.21x | 88.8 % | 89.0 % | 1 | 69.2 % |
| 128 | 4 | 10 | 1.28x | 88.3 % | 87.7 % | 1 | 63.8 % |
| 256 | 1 | 7 | 1.32x | 81.2 % | 81.4 % | 1 | 51.6 % |
| 256 | 4 | 7 | 1.53x | 79.6 % | 79.6 % | 0 | 51.4 % |
| 512 | 1 | 5 | 2.88x | 82.0 % | 82.2 % | 0 | 54.7 % |
| 512 | 4 | 5 | 3.87x | 77.0 % | 77.2 % | 0 | 46.5 % |

Per configuration, bf16 minus fp16 %roof: median -0.13 points (-0.91 to
+0.58), inside what two separate runs differ by. 5 of 52 pairs reach 90 % of
roof, the same five as fp16. Four cells of 364 are slower than Triton, 0.99x,
all D=128 at S=32768, as in fp16.

### Traps

- A define named `BF16` breaks the torch build of *every* variant, fp16
  included: `ATen/Context.h` declares `enum class Float32Precision { ..., BF16 }`.
- `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` accepts `__bf16` vectors only
  through lax vector conversion; with `-flax-vector-conversions=none` it wants
  16 x `short`.

---

## 014 — S read on the device

**Status:** landed, `83415bcc26`.

The op took S as a host int.  The backend inherits Triton's
`AttentionCGSupport.ALWAYS`, so under full CUDA-graph decode that int was
frozen at its capture-time value and every replay attended over the captured
length.  The kernel now reads `seq_lens[0]`.  To keep S off the critical path
the first block's page indices are clamped to the block table's width instead
of S, so the S load and the page-table load go out together.  Cost at S=128:
1-2 % (interleaved A/B, five configurations); none at 32k.  Issuing Q ahead of
the S-dependent early exit measured neutral.  `test_graph_replay_follows_seq_lens`
replays one graph at three lengths.

## 015 — RSPL: row tiles split inside the workgroup

**Status:** landed, `83415bcc26`.

RG splits a kv head's rows over workgroups that each read all of its KV and
rely on L2 to fetch it once.  On `32/2/128` M=4 (rg=4) `GL2C_EA_RDREQ_DRAM`
counts the KV read 1.68 times at S=32768 (calibrated: 1.001x on an rg=1
configuration).  RSPL splits the row tiles over the waves that share a key
tile instead: each loads 1/RSPL of the tile, stages it in LDS
(double-buffered, one barrier per tile), and all read it back; the registers
that frees pay for issuing the next tile before computing this one.  Dev
harness, `32/2/128` M=4, S=32768: 80.1 % (rg=4) -> 91.7 % (rspl=4, nw=4,
nseg=16).  It made a latent race visible: with one tile per workgroup no
merge barrier ordered the waves' rows before `gm_s` was published.

Measured and rejected on the shared tile: two tiles in flight (`PF` on the
shared path) -- 0.74-0.86x as first written, because the conditional issues
and a mid-loop `break` left the waitcnt pass unable to count the pending
loads, so it waited `vmcnt(0)` and drained the prefetch; with unconditional
issues and no break the waits are right (`vmcnt(4)`) and it is still
0.93-1.00x.  Bytes in flight do not limit this path; the per-tile barrier
does.  Loads-only (`ABLATE=64`) on it reaches 88.4 % at 16k, 93.5 % at 32k.

## 016 — Two decompositions per build, switched on the device

**Status:** landed, `9ae12f67da`.

Short and long sequences want different splits of one configuration: row
groups and no merge below ~4k keys, many segments (and RSPL where it applies)
above.  The configuration stays one per `(Hq, Hkv, D, M)`, fixed at graph
capture; the build carries two decompositions and the kernel runs mode B when
the S it reads is `>= SW`.  The source includes itself once per mode
(`mode_a`, `mode_b`; knobs `NSEG RG MINB DSPL RSPL NW PF DOT BFLY GT` and their
`2` versions); the block is launched with the larger mode's waves and a mode
using fewer ends the rest on entry.  `SW=0` builds mode A alone, same
resources as before.

Codegen trap: whichever body sits inside the branch is compiled worse --
2-4 % on mode A there, 15 % on mode B (extra `v_mov` copies; the page-table
load became a vector load behind a `vmcnt(0)`).  Mode B, the long one, falls
through.

`tune.py --split S` descends each mode on its side of S (single-mode builds),
with extra starts one-knob descent cannot reach (rg=1 with every row tile in
RSPL; the dot mode for the short side).

## 017 — PF: a second tile in flight on the unshared path

**Status:** landed, `8cd55c5497`.

Where a second tile's registers fit (small D, small accumulators), the next
tile's loads go out before this one is computed.  Dev harness, contiguous
pages: `16/2/64` M=4 +6.7 % at S=128, +4.7 % at 16k (86.9 -> 91.0 %), +3.4 %
at 32k; `32/8/64` M=4 +4.6 % at S=128.  Where it does not fit it spills and
collapses (`14/2/64` M=4 rg=1: 0.46x; D=128 M=1 needs 53 VGPRs more than
exist), so it is a tuned knob.

Rejected while freeing registers for it: a `sched_barrier` per Q@K chunk
stops the compiler hoisting every LDS operand of the tile (237 -> 220 VGPRs
at D=128 M=1) and costs 1.5-5 %: those loads are what hides LDS latency.

## 018 — The dot decomposition as the short mode

**Status:** landed, `c219ce6083` and `5d1204997f` (rows).

The reference per-q-head dot kernel, measured against the current kernel
(interleaved A/B): 1.04-1.29x at S <= 1024 on all nine D=256/512 M=1
configurations, 0.81-0.96x at 32k.  It is now a mode body (`DOT=1`, with
`BFLY` and `GT`), used as mode A of two-mode builds.  Against the reference:
S and a wave's first block come from the device ahead of time, each later
block is read a tile ahead instead of in front of its own loads, bf16 goes
through `fdot2_f32_bf16`, and only the shipped path is kept (KPW=4, fused
epilogue, last-arriver merge, MSPLIT=1).

matrix.py on the twelve D=256/512 M=1 configurations with their
`--split 4096` rows, against golden/: S=128 1.12-1.34x, S=512 1.01-1.21x,
long contexts unchanged; per configuration 1.013-1.091x.

## 019 — Measured and rejected, same session

| idea | result |
| --- | --- |
| next block's page load after the KV loads (unshared path) | the waitcnt pass then waited for a KV load before storing Q (the load was conditional, so it could not be counted); made unconditional and pinned behind Q it is still 1-3 % slower -- the page-table wait was an L2 hit |
| P@V without P's low half (one WMMA fewer per tile at M=4) | max_rel up to 9.5e-2, and 0.996-1.006x: P@V does not bound the M=4 cells |
| more segments or waves on `16/2/64` M=4, loads only | 84-89 %, worse than fewer: not memory parallelism |

The ceiling behind the remaining long cells: a pure stream (`floor.py`) is at
92.2 % of roof at 16 MiB against 97-98 % at 8-12 MiB and 95.6 % at 32 MiB,
reproducibly, whatever the buffer's alignment.  Most cells still under 90 %
are exactly 16 MiB of KV (Hkv=2 D=128 or Hkv=1 D=256 at 16k, Hkv=2 D=64 at
32k), where a kernel has 2.4 % for everything that is not streaming.
