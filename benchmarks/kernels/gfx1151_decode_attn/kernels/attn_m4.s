	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx1151"
	.amdhsa_code_object_version 6
	.text
	.protected	_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if ; -- Begin function _Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
	.globl	_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
	.p2align	8
	.type	_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if,@function
_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if: ; @_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b256 s[4:11], s[0:1], 0x0
	s_mov_b32 s16, s3
	s_ashr_i32 s17, s3, 31
	v_lshlrev_b32_e32 v1, 1, v0
	s_lshl_b64 s[12:13], s[16:17], 9
	v_dual_mov_b32 v11, 0xff800000 :: v_dual_mov_b32 v10, 0xff800000
	v_dual_mov_b32 v13, 0xff800000 :: v_dual_mov_b32 v12, 0xff800000
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v32, 0 :: v_dual_lshlrev_b32 v17, 2, v0
	v_mov_b32_e32 v16, 0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s4, s12
	s_addc_u32 s5, s5, s13
	v_add_co_u32 v2, s3, s4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v8, null, s5, 0, s3
	v_add_co_u32 v3, vcc_lo, 0x4000, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v4, null, 0, v8, vcc_lo
	v_add_co_u32 v5, vcc_lo, 0x8000, v2
	v_add_co_ci_u32_e64 v6, null, 0, v8, vcc_lo
	v_add_co_u32 v7, vcc_lo, 0xc000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	s_clause 0x3
	global_load_d16_b16 v2, v1, s[4:5]
	global_load_d16_hi_b16 v2, v[3:4], off
	global_load_d16_b16 v3, v[5:6], off
	global_load_d16_hi_b16 v3, v[7:8], off
	s_clause 0x1
	s_load_b64 s[4:5], s[0:1], 0x30
	s_load_b128 s[12:15], s[0:1], 0x20
	v_dual_mov_b32 v4, 0 :: v_dual_mov_b32 v5, 0
	v_dual_mov_b32 v6, 0 :: v_dual_mov_b32 v7, 0
	s_waitcnt vmcnt(3)
	ds_store_b16 v1, v2 offset:16896
	s_waitcnt vmcnt(2)
	ds_store_b16_d16_hi v1, v2 offset:17424
	s_waitcnt vmcnt(1)
	ds_store_b16 v1, v3 offset:17952
	s_waitcnt vmcnt(0)
	ds_store_b16_d16_hi v1, v3 offset:18480
	s_waitcnt lgkmcnt(0)
	s_add_i32 s0, s4, 15
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 28
	s_barrier
	s_add_i32 s0, s0, s1
	buffer_gl0_inv
	s_ashr_i32 s0, s0, 4
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s18, s0, s2
	s_add_i32 s0, s18, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_min_i32 s20, s4, s0
	s_cmp_le_i32 s20, s18
	s_cbranch_scc1 .LBB0_87
; %bb.1:
	s_lshr_b32 s0, s16, 31
	v_or_b32_e32 v2, 0x4200, v1
	s_add_i32 s0, s16, s0
	v_dual_mov_b32 v21, 0 :: v_dual_and_b32 v20, 3, v0
	s_lshl_b32 s0, s0, 7
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v6, 0 :: v_dual_add_nc_u32 v23, 0xffffbe00, v2
	s_and_b32 s0, s0, 0xffffff00
	v_or_b32_e32 v2, -4, v0
	s_ashr_i32 s1, s0, 31
	v_lshrrev_b32_e32 v22, 2, v0
	s_lshl_b64 s[22:23], s[0:1], 1
	v_dual_mov_b32 v14, 0xff800000 :: v_dual_lshlrev_b32 v3, 2, v20
	s_add_u32 s0, s6, s22
	s_addc_u32 s1, s7, s23
	s_add_u32 s3, s8, s22
	v_dual_mov_b32 v36, 0 :: v_dual_add_nc_u32 v27, s4, v2
	s_addc_u32 s4, s9, s23
	s_ashr_i32 s19, s18, 31
	v_add_co_u32 v33, s3, s3, v1
	s_lshl_b64 s[6:7], s[18:19], 13
	v_add_co_ci_u32_e64 v34, null, s4, 0, s3
	s_sub_i32 s4, s20, s18
	s_add_u32 s3, s6, s22
	v_lshl_or_b32 v3, v22, 4, v3
	s_addc_u32 s6, s7, s23
	s_add_u32 s3, s8, s3
	v_add_co_u32 v24, s0, s0, v1
	s_addc_u32 s6, s9, s6
	v_add_co_u32 v8, s3, s3, v1
	v_add_co_ci_u32_e64 v25, null, s1, 0, s0
	v_cmp_gt_u32_e64 s0, 0x80, v0
	v_mul_u32_u24_e32 v26, 0x210, v22
	v_dual_mov_b32 v7, 0 :: v_dual_add_nc_u32 v28, 0x4a40, v3
	v_cmp_gt_u32_e64 s1, 4, v0
	v_dual_mov_b32 v32, 0 :: v_dual_add_nc_u32 v29, 0x4a40, v17
	v_dual_mov_b32 v5, 0 :: v_dual_add_nc_u32 v30, 0x4c50, v17
	v_dual_mov_b32 v18, 0 :: v_dual_add_nc_u32 v31, 0x4c40, v17
	v_mul_u32_u24_e32 v35, 0x210, v20
	v_add_co_ci_u32_e64 v9, null, s6, 0, s3
	v_dual_mov_b32 v15, 0xff800000 :: v_dual_mov_b32 v16, 0
	v_mov_b32_e32 v37, 0xff800000
	v_mov_b32_e32 v3, 0xff800000
	v_mov_b32_e32 v19, 0
	s_branch .LBB0_3
.LBB0_2:                                ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt lgkmcnt(0)
	v_fmac_f32_e32 v4, v36, v40
	v_add_co_u32 v8, vcc_lo, 0x40000, v8
	v_fma_f32 v6, v6, v37, v1
	v_fma_f32 v7, v7, v38, v2
	v_fma_f32 v5, v5, v39, v3
	v_add_co_ci_u32_e64 v9, null, 0, v9, vcc_lo
	v_dual_mov_b32 v15, v12 :: v_dual_mov_b32 v14, v10
	v_mov_b32_e32 v37, v13
	v_dual_mov_b32 v3, v11 :: v_dual_mov_b32 v36, v4
	s_add_i32 s18, s18, 32
	s_sub_i32 s4, s4, 32
	s_cmp_ge_i32 s18, s20
	s_barrier
	buffer_gl0_inv
	s_cbranch_scc1 .LBB0_87
.LBB0_3:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_69 Depth 2
                                        ;     Child Loop BB0_82 Depth 2
                                        ;     Child Loop BB0_86 Depth 2
	v_mov_b16_e32 v1.l, 0
	s_sub_i32 s3, s20, s18
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	s_cmp_gt_i32 s3, 0
	s_cselect_b32 s6, -1, 0
	v_mov_b16_e32 v1.h, v1.l
	s_cmp_lt_i32 s3, 1
	s_cbranch_scc1 .LBB0_5
; %bb.4:                                ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v10, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, s9, v25, vcc_lo
	global_load_d16_hi_b16 v1, v[10:11], off
.LBB0_5:                                ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 2
	s_waitcnt vmcnt(0)
	ds_store_b16_d16_hi v23, v1
	s_cbranch_scc1 .LBB0_7
; %bb.6:                                ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x2000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_b16 v1, v[1:2], off
.LBB0_7:                                ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.h, 0
	s_cmp_lt_i32 s3, 3
	ds_store_b16 v23, v1 offset:528
	v_mov_b16_e32 v2.l, v1.h
	s_cbranch_scc1 .LBB0_9
; %bb.8:                                ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x4000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_9:                                ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 4
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:1056
	s_cbranch_scc1 .LBB0_11
; %bb.10:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x6000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_hi_b16 v1, v[1:2], off
.LBB0_11:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.l, 0
	s_cmp_lt_i32 s3, 5
	ds_store_b16_d16_hi v23, v1 offset:1584
	v_mov_b16_e32 v2.l, v1.l
	s_cbranch_scc1 .LBB0_13
; %bb.12:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x8000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_13:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 6
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:2112
	s_cbranch_scc1 .LBB0_15
; %bb.14:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0xa000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_b16 v1, v[1:2], off
.LBB0_15:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.h, 0
	s_cmp_lt_i32 s3, 7
	ds_store_b16 v23, v1 offset:2640
	v_mov_b16_e32 v2.l, v1.h
	s_cbranch_scc1 .LBB0_17
; %bb.16:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0xc000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_17:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 8
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:3168
	s_cbranch_scc1 .LBB0_19
; %bb.18:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0xe000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_hi_b16 v1, v[1:2], off
.LBB0_19:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.l, 0
	s_cmp_lt_i32 s3, 9
	ds_store_b16_d16_hi v23, v1 offset:3696
	v_mov_b16_e32 v2.l, v1.l
	s_cbranch_scc1 .LBB0_21
; %bb.20:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x10000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_21:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 10
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:4224
	s_cbranch_scc1 .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x12000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_b16 v1, v[1:2], off
.LBB0_23:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.h, 0
	s_cmp_lt_i32 s3, 11
	ds_store_b16 v23, v1 offset:4752
	v_mov_b16_e32 v2.l, v1.h
	s_cbranch_scc1 .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x14000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_25:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 12
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:5280
	s_cbranch_scc1 .LBB0_27
; %bb.26:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x16000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_hi_b16 v1, v[1:2], off
.LBB0_27:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.l, 0
	s_cmp_lt_i32 s3, 13
	ds_store_b16_d16_hi v23, v1 offset:5808
	v_mov_b16_e32 v2.l, v1.l
	s_cbranch_scc1 .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x18000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_29:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 14
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:6336
	s_cbranch_scc1 .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x1a000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_b16 v1, v[1:2], off
.LBB0_31:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.h, 0
	s_cmp_lt_i32 s3, 15
	ds_store_b16 v23, v1 offset:6864
	v_mov_b16_e32 v2.l, v1.h
	s_cbranch_scc1 .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x1c000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_33:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 16
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:7392
	s_cbranch_scc1 .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x1e000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_hi_b16 v1, v[1:2], off
.LBB0_35:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.l, 0
	s_cmp_lt_i32 s3, 17
	ds_store_b16_d16_hi v23, v1 offset:7920
	v_mov_b16_e32 v2.l, v1.l
	s_cbranch_scc1 .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x20000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_37:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 18
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:8448
	s_cbranch_scc1 .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x22000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_b16 v1, v[1:2], off
.LBB0_39:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.h, 0
	s_cmp_lt_i32 s3, 19
	ds_store_b16 v23, v1 offset:8976
	v_mov_b16_e32 v2.l, v1.h
	s_cbranch_scc1 .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x24000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_41:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 20
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:9504
	s_cbranch_scc1 .LBB0_43
; %bb.42:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x26000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_hi_b16 v1, v[1:2], off
.LBB0_43:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.l, 0
	s_cmp_lt_i32 s3, 21
	ds_store_b16_d16_hi v23, v1 offset:10032
	v_mov_b16_e32 v2.l, v1.l
	s_cbranch_scc1 .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x28000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_45:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 22
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:10560
	s_cbranch_scc1 .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x2a000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_b16 v1, v[1:2], off
.LBB0_47:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.h, 0
	s_cmp_lt_i32 s3, 23
	ds_store_b16 v23, v1 offset:11088
	v_mov_b16_e32 v2.l, v1.h
	s_cbranch_scc1 .LBB0_49
; %bb.48:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x2c000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_49:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 24
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:11616
	s_cbranch_scc1 .LBB0_51
; %bb.50:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x2e000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_hi_b16 v1, v[1:2], off
.LBB0_51:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.l, 0
	s_cmp_lt_i32 s3, 25
	ds_store_b16_d16_hi v23, v1 offset:12144
	v_mov_b16_e32 v2.l, v1.l
	s_cbranch_scc1 .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x30000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_53:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 26
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:12672
	s_cbranch_scc1 .LBB0_55
; %bb.54:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x32000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_b16 v1, v[1:2], off
.LBB0_55:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.h, 0
	s_cmp_lt_i32 s3, 27
	ds_store_b16 v23, v1 offset:13200
	v_mov_b16_e32 v2.l, v1.h
	s_cbranch_scc1 .LBB0_57
; %bb.56:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x34000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_57:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 28
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:13728
	s_cbranch_scc1 .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x36000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_hi_b16 v1, v[1:2], off
.LBB0_59:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.l, 0
	s_cmp_lt_i32 s3, 29
	ds_store_b16_d16_hi v23, v1 offset:14256
	v_mov_b16_e32 v2.l, v1.l
	s_cbranch_scc1 .LBB0_61
; %bb.60:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x38000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_61:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 30
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:14784
	s_cbranch_scc1 .LBB0_63
; %bb.62:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x3a000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_b16 v1, v[1:2], off
.LBB0_63:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	v_mov_b16_e32 v1.h, 0
	s_cmp_lt_i32 s3, 31
	ds_store_b16 v23, v1 offset:15312
	v_mov_b16_e32 v2.l, v1.h
	s_cbranch_scc1 .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v2, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v4, null, s9, v25, vcc_lo
	v_add_co_u32 v10, vcc_lo, 0x3c000, v2
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v11, null, 0, v4, vcc_lo
	global_load_d16_b16 v2, v[10:11], off
.LBB0_65:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lt_i32 s3, 32
	s_waitcnt vmcnt(0)
	ds_store_b16 v23, v2 offset:15840
	s_cbranch_scc1 .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_3 Depth=1
	s_ashr_i32 s19, s18, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[18:19], 13
	v_add_co_u32 v1, vcc_lo, v24, s8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s9, v25, vcc_lo
	v_add_co_u32 v1, vcc_lo, 0x3e000, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, 0, v2, vcc_lo
	global_load_d16_hi_b16 v1, v[1:2], off
.LBB0_67:                               ;   in Loop: Header=BB0_3 Depth=1
	s_waitcnt vmcnt(0)
	ds_store_b16_d16_hi v23, v1 offset:16368
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_and_saveexec_b32 s7, s0
	s_cbranch_execz .LBB0_71
; %bb.68:                               ;   in Loop: Header=BB0_3 Depth=1
	v_mov_b32_e32 v1, 0
	s_min_i32 s3, s3, 32
	s_movk_i32 s8, 0xfe00
	s_set_inst_prefetch_distance 0x1
	.p2align	6
.LBB0_69:                               ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_add_nc_u32_e32 v2, s8, v26
	v_add_nc_u32_e32 v4, s8, v35
	s_add_i32 s8, s8, 32
	ds_load_b128 v[10:13], v2 offset:512
	ds_load_b128 v[38:41], v4 offset:17408
	ds_load_b128 v[42:45], v4 offset:17424
	ds_load_b128 v[46:49], v2 offset:528
	s_cmp_eq_u32 s8, 0
	s_waitcnt lgkmcnt(2)
	v_fma_mix_f32 v1, v38, v10, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v38, v10, v1 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v39, v11, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v39, v11, v1 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v40, v12, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v40, v12, v1 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v41, v13, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v41, v13, v1 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_waitcnt lgkmcnt(0)
	v_fma_mix_f32 v1, v42, v46, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v42, v46, v1 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v43, v47, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v43, v47, v1 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v44, v48, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v1, v44, v48, v1 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	v_fma_mix_f32 v1, v45, v49, v1 op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1)
	v_fma_mix_f32 v1, v45, v49, v1 op_sel:[1,1,0] op_sel_hi:[1,1,0]
	s_cbranch_scc0 .LBB0_69
; %bb.70:                               ;   in Loop: Header=BB0_3 Depth=1
	s_set_inst_prefetch_distance 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_dual_mul_f32 v1, s5, v1 :: v_dual_add_nc_u32 v2, s18, v22
	v_cmp_gt_i32_e32 vcc_lo, s3, v22
	v_cmp_le_i32_e64 s3, v2, v27
	s_and_b32 vcc_lo, vcc_lo, s3
	v_cndmask_b32_e32 v1, 0xff800000, v1, vcc_lo
	ds_store_b32 v28, v1
.LBB0_71:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s7
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB0_73
; %bb.72:                               ;   in Loop: Header=BB0_3 Depth=1
	ds_load_2addr_b32 v[1:2], v29 offset1:4
	ds_load_2addr_b32 v[10:11], v29 offset0:8 offset1:12
	ds_load_2addr_b32 v[12:13], v29 offset0:16 offset1:20
	ds_load_2addr_b32 v[38:39], v29 offset0:24 offset1:28
	ds_load_2addr_b32 v[40:41], v29 offset0:32 offset1:36
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v1, 0xff800000, v2
	ds_load_2addr_b32 v[1:2], v29 offset0:40 offset1:44
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v10, v11
	ds_load_2addr_b32 v[10:11], v29 offset0:48 offset1:52
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v12, v13
	ds_load_2addr_b32 v[12:13], v29 offset0:56 offset1:60
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v38, v39
	ds_load_2addr_b32 v[38:39], v29 offset0:64 offset1:68
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v40, v41
	ds_load_2addr_b32 v[40:41], v29 offset0:72 offset1:76
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v1, v2
	ds_load_2addr_b32 v[1:2], v29 offset0:80 offset1:84
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v10, v11
	ds_load_2addr_b32 v[10:11], v29 offset0:88 offset1:92
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v12, v13
	ds_load_2addr_b32 v[12:13], v29 offset0:96 offset1:100
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v38, v39
	ds_load_2addr_b32 v[38:39], v29 offset0:104 offset1:108
	s_waitcnt lgkmcnt(4)
	v_max3_f32 v4, v4, v40, v41
	s_waitcnt lgkmcnt(3)
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_f32 v4, v4, v1, v2
	ds_load_2addr_b32 v[1:2], v29 offset0:112 offset1:116
	s_waitcnt lgkmcnt(3)
	v_max3_f32 v4, v4, v10, v11
	ds_load_2addr_b32 v[10:11], v29 offset0:120 offset1:124
	s_waitcnt lgkmcnt(3)
	v_max3_f32 v4, v4, v12, v13
	s_waitcnt lgkmcnt(2)
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_max3_f32 v4, v4, v38, v39
	s_waitcnt lgkmcnt(1)
	v_max3_f32 v1, v4, v1, v2
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1)
	v_max3_f32 v1, v1, v10, v11
	ds_store_b32 v30, v1
.LBB0_73:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s3
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	ds_load_b128 v[10:13], v21 offset:19536
	v_dual_max_f32 v1, v15, v15 :: v_dual_max_f32 v2, v37, v37
	v_max_f32_e32 v4, v14, v14
	s_waitcnt lgkmcnt(0)
	v_dual_max_f32 v38, v3, v3 :: v_dual_max_f32 v39, v12, v12
	v_dual_max_f32 v10, v10, v10 :: v_dual_max_f32 v11, v11, v11
	v_max_f32_e32 v40, v13, v13
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_max_f32 v12, v1, v10 :: v_dual_max_f32 v13, v2, v11
	v_dual_max_f32 v10, v4, v39 :: v_dual_max_f32 v11, v38, v40
	s_and_saveexec_b32 s3, s0
	s_cbranch_execz .LBB0_77
; %bb.74:                               ;   in Loop: Header=BB0_3 Depth=1
	v_cmp_eq_u32_e32 vcc_lo, 1, v20
	s_mov_b32 s7, exec_lo
	v_cndmask_b32_e32 v1, v12, v13, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 2, v20
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_cndmask_b32_e32 v1, v1, v10, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 3, v20
	v_dual_cndmask_b32 v2, v1, v11 :: v_dual_mov_b32 v1, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_neq_f32_e32 0xff800000, v2
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_3 Depth=1
	ds_load_b32 v1, v28
	s_waitcnt lgkmcnt(0)
	v_sub_f32_e32 v1, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v1, 0x3fb8aa3b, v1
	v_exp_f32_e32 v1, v1
.LBB0_76:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s7
	ds_store_b32 v28, v1
.LBB0_77:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s3
	s_waitcnt lgkmcnt(0)
	s_barrier
	buffer_gl0_inv
	s_and_saveexec_b32 s3, s1
	s_cbranch_execz .LBB0_79
; %bb.78:                               ;   in Loop: Header=BB0_3 Depth=1
	ds_load_2addr_b32 v[1:2], v29 offset1:4
	ds_load_2addr_b32 v[38:39], v29 offset0:8 offset1:12
	ds_load_2addr_b32 v[40:41], v29 offset0:16 offset1:20
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v1, 0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v1, v2
	ds_load_2addr_b32 v[1:2], v29 offset0:24 offset1:28
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v38
	v_add_f32_e32 v4, v4, v39
	ds_load_2addr_b32 v[38:39], v29 offset0:32 offset1:36
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v41
	ds_load_2addr_b32 v[40:41], v29 offset0:40 offset1:44
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v1, v4, v1
	v_add_f32_e32 v4, v1, v2
	ds_load_2addr_b32 v[1:2], v29 offset0:48 offset1:52
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v39
	ds_load_2addr_b32 v[38:39], v29 offset0:56 offset1:60
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v40
	v_add_f32_e32 v4, v4, v41
	ds_load_2addr_b32 v[40:41], v29 offset0:64 offset1:68
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v1, v4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v1, v2
	ds_load_2addr_b32 v[1:2], v29 offset0:72 offset1:76
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v38
	v_add_f32_e32 v4, v4, v39
	ds_load_2addr_b32 v[38:39], v29 offset0:80 offset1:84
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v41
	ds_load_2addr_b32 v[40:41], v29 offset0:88 offset1:92
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v1, v4, v1
	v_add_f32_e32 v4, v1, v2
	ds_load_2addr_b32 v[1:2], v29 offset0:96 offset1:100
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v39
	ds_load_2addr_b32 v[38:39], v29 offset0:104 offset1:108
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v4, v4, v40
	v_add_f32_e32 v4, v4, v41
	ds_load_2addr_b32 v[40:41], v29 offset0:112 offset1:116
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v1, v4, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v1, v2
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v4, v1, v38
	ds_load_2addr_b32 v[1:2], v29 offset0:120 offset1:124
	v_add_f32_e32 v4, v4, v39
	s_waitcnt lgkmcnt(1)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v4, v40
	v_add_f32_e32 v4, v4, v41
	s_waitcnt lgkmcnt(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v4, v1
	v_add_f32_e32 v1, v1, v2
	ds_store_b32 v31, v1
.LBB0_79:                               ;   in Loop: Header=BB0_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s3
	v_dual_sub_f32 v1, v15, v12 :: v_dual_sub_f32 v2, v37, v13
	v_dual_sub_f32 v4, v14, v10 :: v_dual_sub_f32 v3, v3, v11
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v1, 0x3fb8aa3b, v1 :: v_dual_mul_f32 v2, 0x3fb8aa3b, v2
	v_mul_f32_e32 v4, 0x3fb8aa3b, v4
	s_waitcnt lgkmcnt(0)
	s_barrier
	s_delay_alu instid0(VALU_DEP_2)
	v_exp_f32_e32 v14, v1
	v_exp_f32_e32 v15, v2
	v_mul_f32_e32 v1, 0x3fb8aa3b, v3
	v_exp_f32_e32 v39, v4
	buffer_gl0_inv
	s_and_b32 s3, s6, exec_lo
	s_cselect_b32 s3, 1, 0
	v_exp_f32_e32 v40, v1
	ds_load_b128 v[1:4], v21 offset:19520
	s_cmp_lg_u32 s3, 1
	v_cndmask_b32_e32 v37, 0, v14, vcc_lo
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v13
	v_cndmask_b32_e32 v38, 0, v15, vcc_lo
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_mul_f32 v32, v32, v37 :: v_dual_mul_f32 v19, v19, v38
	v_cndmask_b32_e32 v39, 0, v39, vcc_lo
	v_cmp_neq_f32_e32 vcc_lo, 0xff800000, v11
	v_cndmask_b32_e32 v40, 0, v40, vcc_lo
	v_mul_f32_e32 v18, v18, v39
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v16, v16, v40
	s_cbranch_scc1 .LBB0_2
; %bb.80:                               ;   in Loop: Header=BB0_3 Depth=1
	v_med3_i32 v14, s4, 1, 32
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s6, v14
	s_and_b32 s3, s6, 7
	s_cmp_lt_i32 s4, 8
	s_cbranch_scc1 .LBB0_84
; %bb.81:                               ;   in Loop: Header=BB0_3 Depth=1
	v_dual_mov_b32 v15, v9 :: v_dual_mov_b32 v14, v8
	s_and_b32 s6, s6, -8
	s_mov_b32 s7, 0
	s_mov_b32 s8, 0
.LBB0_82:                               ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_i32 s9, s18, s8
	global_load_d16_b16 v73, v[14:15], off
	s_add_i32 s22, s9, 1
	s_add_i32 s24, s9, 2
	s_ashr_i32 s23, s22, 31
	s_add_i32 s26, s9, 3
	s_ashr_i32 s25, s24, 31
	s_lshl_b64 s[22:23], s[22:23], 13
	s_add_i32 s28, s9, 4
	s_ashr_i32 s27, s26, 31
	s_lshl_b64 s[24:25], s[24:25], 13
	v_add_co_u32 v41, vcc_lo, v33, s22
	s_add_i32 s30, s9, 5
	s_ashr_i32 s29, s28, 31
	s_lshl_b64 s[26:27], s[26:27], 13
	v_add_co_ci_u32_e64 v42, null, s23, v34, vcc_lo
	v_add_co_u32 v43, vcc_lo, v33, s24
	s_add_i32 s34, s9, 6
	s_ashr_i32 s31, s30, 31
	s_lshl_b64 s[28:29], s[28:29], 13
	v_add_co_ci_u32_e64 v44, null, s25, v34, vcc_lo
	v_add_co_u32 v45, vcc_lo, v33, s26
	s_add_i32 s36, s9, 7
	s_ashr_i32 s35, s34, 31
	s_lshl_b64 s[30:31], s[30:31], 13
	v_add_co_ci_u32_e64 v46, null, s27, v34, vcc_lo
	v_add_co_u32 v47, vcc_lo, v33, s28
	s_ashr_i32 s37, s36, 31
	s_lshl_b64 s[34:35], s[34:35], 13
	v_add_co_ci_u32_e64 v48, null, s29, v34, vcc_lo
	v_add_co_u32 v49, vcc_lo, v33, s30
	s_lshl_b64 s[36:37], s[36:37], 13
	v_add_co_ci_u32_e64 v50, null, s31, v34, vcc_lo
	v_add_co_u32 v51, vcc_lo, v33, s34
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v52, null, s35, v34, vcc_lo
	v_add_co_u32 v53, vcc_lo, v33, s36
	v_add_co_ci_u32_e64 v54, null, s37, v34, vcc_lo
	s_clause 0x6
	global_load_d16_b16 v74, v[41:42], off
	global_load_d16_b16 v75, v[43:44], off
	global_load_d16_b16 v76, v[45:46], off
	global_load_d16_b16 v77, v[47:48], off
	global_load_d16_b16 v78, v[49:50], off
	global_load_d16_b16 v79, v[51:52], off
	global_load_d16_b16 v80, v[53:54], off
	v_mov_b32_e32 v69, s7
	ds_load_b128 v[41:44], v69 offset:19008
	ds_load_b128 v[45:48], v69 offset:19024
	ds_load_b128 v[49:52], v69 offset:19040
	ds_load_b128 v[53:56], v69 offset:19056
	ds_load_b128 v[57:60], v69 offset:19072
	ds_load_b128 v[61:64], v69 offset:19088
	ds_load_b128 v[65:68], v69 offset:19104
	ds_load_b128 v[69:72], v69 offset:19120
	v_add_co_u32 v14, vcc_lo, 0x10000, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v15, null, 0, v15, vcc_lo
	s_add_i32 s8, s8, 8
	s_addk_i32 s7, 0x80
	s_cmp_eq_u32 s6, s8
	s_waitcnt vmcnt(7) lgkmcnt(7)
	v_fma_mix_f32 v32, v41, v73, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v42, v73, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v43, v73, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v44, v73, v16 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(6) lgkmcnt(6)
	v_fma_mix_f32 v32, v45, v74, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v46, v74, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v47, v74, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v48, v74, v16 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(5) lgkmcnt(5)
	v_fma_mix_f32 v32, v49, v75, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v50, v75, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v51, v75, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v52, v75, v16 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(4) lgkmcnt(4)
	v_fma_mix_f32 v32, v53, v76, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v54, v76, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v55, v76, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v56, v76, v16 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(3) lgkmcnt(3)
	v_fma_mix_f32 v32, v57, v77, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v58, v77, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v59, v77, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v60, v77, v16 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(2) lgkmcnt(2)
	v_fma_mix_f32 v32, v61, v78, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v62, v78, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v63, v78, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v64, v78, v16 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(1) lgkmcnt(1)
	v_fma_mix_f32 v32, v65, v79, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v66, v79, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v67, v79, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v68, v79, v16 op_sel_hi:[0,1,0]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_fma_mix_f32 v32, v69, v80, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v70, v80, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v71, v80, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v72, v80, v16 op_sel_hi:[0,1,0]
	s_cbranch_scc0 .LBB0_82
; %bb.83:                               ;   in Loop: Header=BB0_3 Depth=1
	s_cmp_lg_u32 s3, 0
	s_cselect_b32 s7, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s7
	s_cbranch_vccz .LBB0_2
	s_branch .LBB0_85
.LBB0_84:                               ;   in Loop: Header=BB0_3 Depth=1
	s_mov_b32 s6, 0
	s_cbranch_execz .LBB0_2
.LBB0_85:                               ;   in Loop: Header=BB0_3 Depth=1
	s_mov_b32 s7, 0
	.p2align	6
.LBB0_86:                               ;   Parent Loop BB0_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_add_i32 s8, s6, s18
	s_add_i32 s7, s7, 1
	s_ashr_i32 s9, s8, 31
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshl_b64 s[8:9], s[8:9], 13
	v_add_co_u32 v14, vcc_lo, v33, s8
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v15, null, s9, v34, vcc_lo
	s_lshl_b32 s8, s6, 4
	s_add_i32 s6, s6, 1
	s_cmp_lg_u32 s7, s3
	global_load_d16_b16 v14, v[14:15], off
	v_mov_b32_e32 v15, s8
	ds_load_b128 v[41:44], v15 offset:19008
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_fma_mix_f32 v32, v41, v14, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v19, v42, v14, v19 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v18, v43, v14, v18 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v16, v44, v14, v16 op_sel_hi:[0,1,0]
	s_cbranch_scc1 .LBB0_86
	s_branch .LBB0_2
.LBB0_87:
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[0:1], s[16:17], 6
	s_lshl_b64 s[2:3], s[2:3], 2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_u32 s0, s0, s2
	v_add_co_u32 v1, s2, s10, v17
	s_addc_u32 s1, s1, s3
	v_add_co_ci_u32_e64 v2, null, s11, 0, s2
	s_lshl_b64 s[2:3], s[0:1], 10
	v_add_co_u32 v1, vcc_lo, v1, s2
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v2, null, s3, v2, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 0, v0
	global_store_b32 v[1:2], v32, off
	s_and_saveexec_b32 s2, vcc_lo
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execz .LBB0_89
; %bb.88:
	global_store_b32 v[1:2], v19, off offset:1024
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr6
.LBB0_89:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB0_91
; %bb.90:
	s_lshl_b64 s[4:5], s[0:1], 2
	v_mov_b32_e32 v0, 0
	s_add_u32 s6, s14, s4
	s_addc_u32 s7, s15, s5
	s_add_u32 s4, s12, s4
	s_addc_u32 s5, s13, s5
	global_store_b32 v[1:2], v19, off offset:1024
	s_clause 0x1
	global_store_b64 v0, v[12:13], s[4:5]
	global_store_b64 v0, v[6:7], s[6:7]
.LBB0_91:
	s_or_b32 exec_lo, exec_lo, s2
	global_store_b32 v[1:2], v18, off offset:2048
	s_and_saveexec_b32 s2, vcc_lo
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s2, exec_lo, s2
	s_cbranch_execnz .LBB0_94
; %bb.92:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execnz .LBB0_95
.LBB0_93:
	s_endpgm
.LBB0_94:
	global_store_b32 v[1:2], v16, off offset:3072
                                        ; implicit-def: $vgpr1_vgpr2
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr5
                                        ; implicit-def: $vgpr4
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB0_93
.LBB0_95:
	s_lshl_b64 s[0:1], s[0:1], 2
	v_mov_b32_e32 v0, 0
	s_add_u32 s2, s14, s0
	s_addc_u32 s3, s15, s1
	v_mov_b32_e32 v6, v4
	s_add_u32 s0, s12, s0
	s_addc_u32 s1, s13, s1
	global_store_b32 v[1:2], v16, off offset:3072
	s_clause 0x1
	global_store_b64 v0, v[10:11], s[0:1] offset:8
	global_store_b64 v0, v[5:6], s[2:3] offset:8
	s_endpgm
.Lfunc_end0:
	.size	_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if, .Lfunc_end0-_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
		.amdhsa_group_segment_fixed_size 19552
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 56
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 97
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end0-_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.num_vgpr, 81
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.num_agpr, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.numbered_sgpr, 38
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.num_named_barrier, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.private_seg_size, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.uses_vcc, 1
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.uses_flat_scratch, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.has_dyn_sized_stack, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.has_recursion, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6212
; TotalNumSgprs: 40
; NumVgprs: 81
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 19552 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 97
; Occupancy: 12
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	_Z15reduce_segmentsPKfS0_S0_Pf ; -- Begin function _Z15reduce_segmentsPKfS0_S0_Pf
	.globl	_Z15reduce_segmentsPKfS0_S0_Pf
	.p2align	8
	.type	_Z15reduce_segmentsPKfS0_S0_Pf,@function
_Z15reduce_segmentsPKfS0_S0_Pf:         ; @_Z15reduce_segmentsPKfS0_S0_Pf
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_b256 s[4:11], s[0:1], 0x0
	s_mov_b32 s12, s3
	s_ashr_i32 s13, s3, 31
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[0:1], s[12:13], 8
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v2, 0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s6, s0
	s_addc_u32 s7, s7, s1
	s_lshl_b64 s[0:1], s[2:3], 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_u32 s6, s6, s0
	s_addc_u32 s7, s7, s1
	s_clause 0xf
	s_load_b32 s30, s[6:7], 0x10
	s_load_b32 s0, s[6:7], 0x0
	s_load_b32 s29, s[6:7], 0x20
	s_load_b32 s28, s[6:7], 0x30
	s_load_b32 s27, s[6:7], 0x40
	s_load_b32 s26, s[6:7], 0x50
	s_load_b32 s25, s[6:7], 0x60
	s_load_b32 s24, s[6:7], 0x70
	s_load_b32 s23, s[6:7], 0x80
	s_load_b32 s22, s[6:7], 0x90
	s_load_b32 s21, s[6:7], 0xa0
	s_load_b32 s20, s[6:7], 0xb0
	s_load_b32 s19, s[6:7], 0xc0
	s_load_b32 s18, s[6:7], 0xd0
	s_load_b32 s17, s[6:7], 0xe0
	s_load_b32 s14, s[6:7], 0xf0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_f32 v1, s0, 0xff800000, v1
	v_max3_f32 v1, v1, s29, s28
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_f32 v1, v1, s27, s26
	v_max3_f32 v1, v1, s25, s24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_f32 v1, v1, s23, s22
	v_max3_f32 v1, v1, s21, s20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_f32 v1, v1, s19, s18
	v_max3_f32 v1, v1, s17, s14
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s15, v1
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v1
	s_cmp_neq_f32 s15, 0xff800000
	s_cselect_b32 s16, -1, 0
	s_cbranch_vccnz .LBB1_2
; %bb.1:
	s_sub_f32 s0, s0, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s0, s0, 0x3fb8aa3b
	v_exp_f32_e32 v4, s0
.LBB1_2:
	s_lshl_b64 s[0:1], s[12:13], 6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	s_add_u32 s6, s0, s2
	s_addc_u32 s7, s1, s3
	s_lshl_b64 s[0:1], s[6:7], 2
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	s_add_u32 s0, s8, s0
	s_addc_u32 s1, s9, s1
	s_and_b32 s8, s16, exec_lo
	s_cselect_b32 s8, 1, 0
	s_cmp_lg_u32 s8, 1
	s_cbranch_scc1 .LBB1_4
; %bb.3:
	s_sub_f32 s8, s30, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s8, s8, 0x3fb8aa3b
	v_exp_f32_e32 v2, s8
.LBB1_4:
	s_and_b32 s8, s16, exec_lo
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_cselect_b32 s8, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, 1
	s_cbranch_scc1 .LBB1_6
; %bb.5:
	s_sub_f32 s8, s29, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s8, s8, 0x3fb8aa3b
	v_exp_f32_e32 v6, s8
.LBB1_6:
	s_and_b32 s8, s16, exec_lo
	s_cselect_b32 s8, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, 1
	s_cbranch_scc1 .LBB1_8
; %bb.7:
	s_sub_f32 s8, s28, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s8, s8, 0x3fb8aa3b
	v_exp_f32_e32 v5, s8
.LBB1_8:
	v_dual_mov_b32 v8, 0 :: v_dual_lshlrev_b32 v3, 2, v0
	s_and_b32 s8, s16, exec_lo
	v_mov_b32_e32 v7, 0
	s_cselect_b32 s8, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, 1
	s_cbranch_scc1 .LBB1_10
; %bb.9:
	s_sub_f32 s8, s27, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s8, s8, 0x3fb8aa3b
	v_exp_f32_e32 v8, s8
.LBB1_10:
	s_and_b32 s8, s16, exec_lo
	s_cselect_b32 s8, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, 1
	s_cbranch_scc1 .LBB1_12
; %bb.11:
	s_sub_f32 s8, s26, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s8, s8, 0x3fb8aa3b
	v_exp_f32_e32 v7, s8
.LBB1_12:
	s_and_b32 s8, s16, exec_lo
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v10, 0
	s_cselect_b32 s8, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, 1
	s_cbranch_scc1 .LBB1_14
; %bb.13:
	s_sub_f32 s8, s25, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s8, s8, 0x3fb8aa3b
	v_exp_f32_e32 v10, s8
.LBB1_14:
	v_add_co_u32 v0, s4, s4, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v1, null, s5, 0, s4
	s_and_b32 s4, s16, exec_lo
	s_cselect_b32 s4, 1, 0
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_16
; %bb.15:
	s_sub_f32 s4, s24, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s4, s4, 0x3fb8aa3b
	v_exp_f32_e32 v9, s4
.LBB1_16:
	s_lshl_b64 s[4:5], s[6:7], 10
	s_and_b32 s6, s16, exec_lo
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	s_cselect_b32 s6, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_cbranch_scc1 .LBB1_18
; %bb.17:
	s_sub_f32 s6, s23, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s6, s6, 0x3fb8aa3b
	v_exp_f32_e32 v12, s6
.LBB1_18:
	s_and_b32 s6, s16, exec_lo
	s_cselect_b32 s6, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s6, 1
	s_cbranch_scc1 .LBB1_20
; %bb.19:
	s_sub_f32 s6, s22, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s6, s6, 0x3fb8aa3b
	v_exp_f32_e32 v11, s6
.LBB1_20:
	v_add_co_u32 v0, vcc_lo, v0, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	s_and_b32 s4, s16, exec_lo
	v_dual_mov_b32 v13, 0 :: v_dual_mov_b32 v14, 0
	s_cselect_b32 s4, 1, 0
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_22
; %bb.21:
	s_sub_f32 s4, s21, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s4, s4, 0x3fb8aa3b
	v_exp_f32_e32 v14, s4
.LBB1_22:
	s_and_b32 s4, s16, exec_lo
	s_cselect_b32 s4, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_24
; %bb.23:
	s_sub_f32 s4, s20, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s4, s4, 0x3fb8aa3b
	v_exp_f32_e32 v13, s4
.LBB1_24:
	s_and_b32 s4, s16, exec_lo
	v_dual_mov_b32 v15, 0 :: v_dual_mov_b32 v16, 0
	s_cselect_b32 s4, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_26
; %bb.25:
	s_sub_f32 s4, s19, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s4, s4, 0x3fb8aa3b
	v_exp_f32_e32 v16, s4
.LBB1_26:
	s_and_b32 s4, s16, exec_lo
	s_cselect_b32 s4, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_28
; %bb.27:
	s_sub_f32 s4, s18, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s4, s4, 0x3fb8aa3b
	v_exp_f32_e32 v15, s4
.LBB1_28:
	s_and_b32 s4, s16, exec_lo
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v18, 0
	s_cselect_b32 s4, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_30
; %bb.29:
	s_sub_f32 s4, s17, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s4, s4, 0x3fb8aa3b
	v_exp_f32_e32 v18, s4
.LBB1_30:
	v_add_co_u32 v20, vcc_lo, v0, 0x2000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v21, null, 0, v1, vcc_lo
	v_add_co_u32 v22, vcc_lo, v0, 0x4000
	v_add_co_ci_u32_e64 v23, null, 0, v1, vcc_lo
	v_add_co_u32 v28, vcc_lo, v0, 0x6000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v29, null, 0, v1, vcc_lo
	v_add_co_u32 v30, vcc_lo, v0, 0x8000
	v_add_co_ci_u32_e64 v31, null, 0, v1, vcc_lo
	s_clause 0x7
	global_load_b32 v27, v[20:21], off offset:-4096
	global_load_b32 v26, v[20:21], off
	global_load_b32 v25, v[22:23], off offset:-4096
	global_load_b32 v24, v[22:23], off
	global_load_b32 v21, v[28:29], off offset:-4096
	global_load_b32 v20, v[28:29], off
	global_load_b32 v22, v[30:31], off offset:-4096
	global_load_b32 v23, v[30:31], off
	v_add_co_u32 v28, vcc_lo, v0, 0xa000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v29, null, 0, v1, vcc_lo
	v_add_co_u32 v30, vcc_lo, v0, 0xc000
	v_add_co_ci_u32_e64 v31, null, 0, v1, vcc_lo
	v_add_co_u32 v34, vcc_lo, v0, 0xe000
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v35, null, 0, v1, vcc_lo
	s_clause 0x6
	global_load_b32 v33, v[28:29], off offset:-4096
	global_load_b32 v32, v[28:29], off
	global_load_b32 v29, v[30:31], off offset:-4096
	global_load_b32 v28, v[30:31], off
	global_load_b32 v30, v[34:35], off offset:-4096
	global_load_b32 v31, v[34:35], off
	global_load_b32 v19, v[0:1], off
	s_clause 0xe
	s_load_b32 s25, s[0:1], 0x0
	s_load_b32 s24, s[0:1], 0x10
	s_load_b32 s23, s[0:1], 0x20
	s_load_b32 s22, s[0:1], 0x30
	s_load_b32 s21, s[0:1], 0x40
	s_load_b32 s20, s[0:1], 0x50
	s_load_b32 s19, s[0:1], 0x60
	s_load_b32 s18, s[0:1], 0x70
	s_load_b32 s17, s[0:1], 0x80
	s_load_b32 s9, s[0:1], 0x90
	s_load_b32 s8, s[0:1], 0xa0
	s_load_b32 s7, s[0:1], 0xb0
	s_load_b32 s6, s[0:1], 0xc0
	s_load_b32 s5, s[0:1], 0xd0
	s_load_b32 s4, s[0:1], 0xe0
	s_and_b32 s16, s16, exec_lo
	s_cselect_b32 s16, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s16, 1
	s_cbranch_scc1 .LBB1_32
; %bb.31:
	s_sub_f32 s14, s14, s15
	s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)
	s_mul_f32 s14, s14, 0x3fb8aa3b
	v_exp_f32_e32 v17, s14
.LBB1_32:
	v_add_co_u32 v0, vcc_lo, 0xf000, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_load_b32 s0, s[0:1], 0xf0
	global_load_b32 v0, v[0:1], off
	s_waitcnt lgkmcnt(0)
	v_fma_f32 v1, v4, s25, 0
	s_waitcnt vmcnt(1)
	v_fma_f32 v4, v4, v19, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v1, s24, v2 :: v_dual_fmac_f32 v4, v2, v27
	v_fmac_f32_e32 v4, v6, v26
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v1, s23, v6 :: v_dual_fmac_f32 v4, v5, v25
	v_dual_fmac_f32 v1, s22, v5 :: v_dual_fmac_f32 v4, v8, v24
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v1, s21, v8 :: v_dual_fmac_f32 v4, v7, v21
	v_dual_fmac_f32 v1, s20, v7 :: v_dual_fmac_f32 v4, v10, v20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, s19, v10
	v_dual_fmac_f32 v4, v9, v22 :: v_dual_fmac_f32 v1, s18, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v4, v12, v23 :: v_dual_fmac_f32 v1, s17, v12
	v_dual_fmac_f32 v4, v11, v33 :: v_dual_fmac_f32 v1, s9, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v4, v14, v32 :: v_dual_fmac_f32 v1, s8, v14
	v_fmac_f32_e32 v4, v13, v29
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v1, s7, v13 :: v_dual_fmac_f32 v4, v16, v28
	v_dual_fmac_f32 v1, s6, v16 :: v_dual_fmac_f32 v4, v15, v30
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v1, s5, v15
	v_dual_fmac_f32 v4, v18, v31 :: v_dual_fmac_f32 v1, s4, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_fmac_f32_e32 v1, s0, v17
	s_lshl_b64 s[0:1], s[2:3], 15
	s_add_u32 s2, s10, s0
	s_addc_u32 s3, s11, s1
	s_lshl_b64 s[0:1], s[12:13], 10
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v4, v17, v0
	v_div_scale_f32 v0, null, v1, v1, v4
	v_div_scale_f32 v6, vcc_lo, v4, v1, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_f32_e32 v2, v0
	v_fma_f32 v5, -v0, v2, 1.0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v2, v5, v2
	v_mul_f32_e32 v5, v6, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v7, -v0, v5, v6
	v_fmac_f32_e32 v5, v7, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v0, -v0, v5, v6
	v_div_fmas_f32 v0, v0, v2, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_div_fixup_f32 v0, v0, v1, v4
	global_store_b32 v3, v0, s[0:1]
	s_endpgm
.Lfunc_end1:
	.size	_Z15reduce_segmentsPKfS0_S0_Pf, .Lfunc_end1-_Z15reduce_segmentsPKfS0_S0_Pf
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z15reduce_segmentsPKfS0_S0_Pf
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 36
		.amdhsa_next_free_sgpr 31
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_shared_vgpr_count 0
		.amdhsa_inst_pref_size ((instprefsize(.Lfunc_end1-_Z15reduce_segmentsPKfS0_S0_Pf)<<4)&1008)>>4
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.num_vgpr, 36
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.num_agpr, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.numbered_sgpr, 31
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.num_named_barrier, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.private_seg_size, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.uses_vcc, 1
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.uses_flat_scratch, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.has_dyn_sized_stack, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.has_recursion, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1824
; TotalNumSgprs: 33
; NumVgprs: 36
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 4
; NumSGPRsForWavesPerEU: 33
; NumVGPRsForWavesPerEU: 36
; Occupancy: 16
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.type	__hip_cuid_98ed3a62a1333e4e,@object ; @__hip_cuid_98ed3a62a1333e4e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_98ed3a62a1333e4e
__hip_cuid_98ed3a62a1333e4e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_98ed3a62a1333e4e, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 0bace1908348b840e6aa1b4b6e12151dae208158)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_98ed3a62a1333e4e
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 19552
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         _Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     81
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .gfx1250_revision: B0
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z15reduce_segmentsPKfS0_S0_Pf
    .private_segment_fixed_size: 0
    .sgpr_count:     33
    .sgpr_spill_count: 0
    .symbol:         _Z15reduce_segmentsPKfS0_S0_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     36
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
