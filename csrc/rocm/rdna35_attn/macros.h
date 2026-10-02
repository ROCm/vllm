// Helpers for the X-macro list of decode-attention variants (variants.def).
//
// An entry is
//
//   RDNA35_VARIANT(group, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS, MAXM, BS,
//                  LAYOUT, NSEG, RG, MINB, NW, DSPL, RSPL, PF, CPUB, VINLDS,
//                  BATCH, DOT, BFLY, GT, WIN, KV_BF16)
//
// with every field an integer literal, exactly the defines
// rdna35_decode_attn.cu is built with (DSPL 0 is the kernel's own rule).
// `group` names the translation unit the variant is compiled in.
#pragma once

#define RDNA35_STRIP(...) __VA_ARGS__
// m applied to the fields of a parenthesised entry: RDNA35_CALL(m, (a, b))
// is m(a, b), with the entry expanded before m collects its arguments.
#define RDNA35_CALL(m, args) m args
#define RDNA35_PREPEND(p, entry) (p, RDNA35_STRIP entry)

// A C identifier unique to the variant, from every field but the group:
// RDNA35_ID(launch, g, 128, 16, 2, ...) is launch_d128_q16_kv2_...
#define RDNA35_ID(p, g, d, hq, hkv, m, bs, l, n, rg, mb, nw, ds, rs, pf, cp, \
                  vl, b, dot, bf, gt, win, t)                                \
  p##_d##d##_q##hq##_kv##hkv##_m##m##_bs##bs##_l##l##_n##n##_rg##rg##_mb##mb##_w##nw##_ds##ds##_rs##rs##_pf##pf##_cp##cp##_vl##vl##_b##b##_dot##dot##_bf##bf##_gt##gt##_win##win##_t##t

// The key a variant is looked up by: every field but the group, in order.
#define RDNA35_KEY(g, ...) __VA_ARGS__
#define RDNA35_NFIELDS 21

// Field i (1-based, after the group) of a parenthesised entry.
#define RDNA35_FIELD(i, entry) RDNA35_CALL(RDNA35_F##i, entry)
#define RDNA35_F1(g, a, ...) a
#define RDNA35_F2(g, a, b, ...) b
#define RDNA35_F3(g, a, b, c, ...) c
#define RDNA35_F4(g, a, b, c, d, ...) d
#define RDNA35_F5(g, a, b, c, d, e, ...) e
#define RDNA35_F6(g, a, b, c, d, e, f, ...) f
#define RDNA35_F7(g, a, b, c, d, e, f, h, ...) h
#define RDNA35_F8(g, a, b, c, d, e, f, h, i, ...) i
#define RDNA35_F9(g, a, b, c, d, e, f, h, i, j, ...) j
#define RDNA35_F10(g, a, b, c, d, e, f, h, i, j, k, ...) k
#define RDNA35_F11(g, a, b, c, d, e, f, h, i, j, k, l, ...) l
#define RDNA35_F12(g, a, b, c, d, e, f, h, i, j, k, l, m, ...) m
#define RDNA35_F13(g, a, b, c, d, e, f, h, i, j, k, l, m, n, ...) n
#define RDNA35_F14(g, a, b, c, d, e, f, h, i, j, k, l, m, n, o, ...) o
#define RDNA35_F15(g, a, b, c, d, e, f, h, i, j, k, l, m, n, o, p, ...) p
#define RDNA35_F16(g, a, b, c, d, e, f, h, i, j, k, l, m, n, o, p, q, ...) q
#define RDNA35_F17(g, a, b, c, d, e, f, h, i, j, k, l, m, n, o, p, q, r, ...) r
#define RDNA35_F18(g, a, b, c, d, e, f, h, i, j, k, l, m, n, o, p, q, r, s, \
                   ...)                                                     \
  s
#define RDNA35_F19(g, a, b, c, d, e, f, h, i, j, k, l, m, n, o, p, q, r, s, t, \
                   ...)                                                        \
  t
#define RDNA35_F20(g, a, b, c, d, e, f, h, i, j, k, l, m, n, o, p, q, r, s, t, \
                   u, ...)                                                     \
  u
#define RDNA35_F21(g, a, b, c, d, e, f, h, i, j, k, l, m, n, o, p, q, r, s, t, \
                   u, v)                                                       \
  v
