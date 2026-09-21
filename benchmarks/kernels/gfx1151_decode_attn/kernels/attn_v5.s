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
	s_clause 0x1
	s_load_b64 s[30:31], s[0:1], 0x30
	s_load_b128 s[24:27], s[0:1], 0x20
	v_lshrrev_b32_e32 v51, 5, v0
	s_load_b256 s[16:23], s[0:1], 0x0
	s_lshl_b32 s33, s2, 3
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v49, 0xff800000
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_or_b32_e32 v1, s33, v51
	v_dual_mov_b32 v55, 0 :: v_dual_and_b32 v50, 31, v0
	v_dual_mov_b32 v65, 0xff800000 :: v_dual_mov_b32 v66, 0xff800000
	v_dual_mov_b32 v58, 0 :: v_dual_lshlrev_b32 v53, 2, v1
	v_dual_mov_b32 v67, 0xff800000 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, v54
	v_dual_mov_b32 v47, v54 :: v_dual_lshlrev_b32 v52, 3, v50
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_i32_e32 vcc_lo, s30, v53
	v_mov_b32_e32 v46, v54
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, v54
	v_mov_b32_e32 v42, v54
	v_dual_mov_b32 v43, v54 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v37, v54 :: v_dual_mov_b32 v32, 0
	v_mov_b32_e32 v38, v54
	v_dual_mov_b32 v39, v54 :: v_dual_mov_b32 v12, 0
	v_dual_mov_b32 v33, v54 :: v_dual_mov_b32 v8, 0
	v_mov_b32_e32 v34, v54
	v_dual_mov_b32 v35, v54 :: v_dual_mov_b32 v4, 0
	v_dual_mov_b32 v13, v54 :: v_dual_mov_b32 v0, 0
	v_mov_b32_e32 v14, v54
	v_mov_b32_e32 v15, v54
	v_mov_b32_e32 v9, v54
	v_mov_b32_e32 v10, v54
	v_mov_b32_e32 v11, v54
	v_mov_b32_e32 v5, v54
	v_mov_b32_e32 v6, v54
	v_mov_b32_e32 v7, v54
	v_mov_b32_e32 v1, v54
	v_mov_b32_e32 v2, v54
	v_mov_b32_e32 v3, v54
	s_mov_b32 s28, s3
	s_ashr_i32 s29, s3, 31
	s_and_saveexec_b32 s34, vcc_lo
	s_cbranch_execz .LBB0_4
; %bb.1:
	s_lshr_b32 s0, s28, 31
	v_lshlrev_b32_e32 v6, 1, v52
	s_add_i32 s2, s28, s0
	s_lshl_b64 s[0:1], s[28:29], 9
	s_add_i32 s35, s30, -4
	s_ashr_i32 s2, s2, 1
	s_add_u32 s0, s16, s0
	s_addc_u32 s1, s17, s1
	v_add_co_u32 v4, s3, s0, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v5, null, s1, 0, s3
	s_ashr_i32 s3, s2, 31
	v_add_co_u32 v0, vcc_lo, 0x4000, v4
	v_add_co_ci_u32_e64 v1, null, 0, v5, vcc_lo
	v_add_co_u32 v2, vcc_lo, 0x8000, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v3, null, 0, v5, vcc_lo
	v_add_co_u32 v4, vcc_lo, 0xc000, v4
	v_add_co_ci_u32_e64 v5, null, 0, v5, vcc_lo
	s_clause 0x3
	global_load_b128 v[16:19], v6, s[0:1]
	global_load_b128 v[20:23], v[0:1], off
	global_load_b128 v[24:27], v[2:3], off
	global_load_b128 v[28:31], v[4:5], off
	v_mbcnt_lo_u32_b32 v0, -1, 0
	s_lshl_b64 s[0:1], s[2:3], 8
	v_mov_b32_e32 v3, 0
	s_mul_f32 s16, s31, 0x3fb8aa3b
	v_or_b32_e32 v59, s0, v52
	v_xor_b32_e32 v1, 1, v0
	v_xor_b32_e32 v2, 2, v0
	v_xor_b32_e32 v4, 4, v0
	v_xor_b32_e32 v5, 8, v0
	v_xor_b32_e32 v6, 16, v0
	v_cmp_gt_u32_e32 vcc_lo, 32, v1
	v_dual_mov_b32 v7, v3 :: v_dual_mov_b32 v66, 0xff800000
	v_mov_b32_e32 v11, v3
	v_dual_mov_b32 v10, v3 :: v_dual_cndmask_b32 v1, v0, v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v2
	v_mov_b32_e32 v57, s1
	v_mov_b32_e32 v9, v3
	v_dual_mov_b32 v67, 0xff800000 :: v_dual_mov_b32 v8, v3
	v_cndmask_b32_e32 v2, v0, v2, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v4
	v_mov_b32_e32 v15, v3
	v_mov_b32_e32 v14, v3
	v_mov_b32_e32 v13, v3
	v_dual_mov_b32 v2, v3 :: v_dual_lshlrev_b32 v61, 2, v2
	v_cndmask_b32_e32 v4, v0, v4, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, 32, v5
	v_mov_b32_e32 v12, v3
	v_mov_b32_e32 v35, v3
	v_mov_b32_e32 v34, v3
	v_dual_cndmask_b32 v5, v0, v5 :: v_dual_lshlrev_b32 v62, 2, v4
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	v_dual_mov_b32 v1, v3 :: v_dual_lshlrev_b32 v60, 2, v1
	v_mov_b32_e32 v4, v3
	v_dual_mov_b32 v33, v3 :: v_dual_cndmask_b32 v0, v0, v6
	v_dual_mov_b32 v6, v3 :: v_dual_mov_b32 v49, 0xff800000
	v_mov_b32_e32 v32, v3
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v39, v3 :: v_dual_lshlrev_b32 v64, 2, v0
	v_dual_mov_b32 v0, v3 :: v_dual_lshlrev_b32 v63, 2, v5
	v_mov_b32_e32 v65, 0xff800000
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v38, v3
	v_mov_b32_e32 v37, v3
	v_mov_b32_e32 v36, v3
	v_mov_b32_e32 v43, v3
	v_mov_b32_e32 v42, v3
	v_mov_b32_e32 v41, v3
	v_mov_b32_e32 v40, v3
	v_mov_b32_e32 v47, v3
	v_mov_b32_e32 v46, v3
	v_mov_b32_e32 v45, v3
	v_mov_b32_e32 v44, v3
	v_mov_b32_e32 v58, v3
	v_mov_b32_e32 v56, v3
	v_mov_b32_e32 v55, v3
	v_mov_b32_e32 v54, v3
	s_mov_b32 s17, 0
	s_add_i32 s31, s30, -1
	s_add_i32 s36, s30, -3
	s_add_i32 s37, s30, -2
	s_mov_b32 s38, s16
	s_mov_b32 s39, s16
	s_mov_b32 s40, s16
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	v_min_i32_e32 v48, s31, v53
	v_dual_mov_b32 v70, 0 :: v_dual_add_nc_u32 v79, 1, v53
	v_dual_mov_b32 v69, 0 :: v_dual_add_nc_u32 v68, 2, v53
	v_mov_b32_e32 v90, v49
	s_delay_alu instid0(VALU_DEP_4)
	v_ashrrev_i32_e32 v49, 31, v48
	v_dual_mov_b32 v83, 0 :: v_dual_add_nc_u32 v72, 3, v53
	v_dual_mov_b32 v89, v66 :: v_dual_mov_b32 v88, v65
	v_min_i32_e32 v65, s31, v79
	v_min_i32_e32 v91, s31, v68
	v_lshlrev_b64 v[48:49], 12, v[48:49]
	v_min_i32_e32 v93, s31, v72
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v74, 0
	v_ashrrev_i32_e32 v66, 31, v65
	v_ashrrev_i32_e32 v92, 31, v91
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_ashrrev_i32_e32 v94, 31, v93
	v_add_co_u32 v48, s10, v48, v59
	v_lshlrev_b64 v[65:66], 12, v[65:66]
	v_add_co_ci_u32_e64 v49, null, v49, v57, s10
	v_lshlrev_b64 v[91:92], 12, v[91:92]
	v_lshlrev_b64 v[93:94], 12, v[93:94]
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v76, 0
	v_add_co_u32 v65, s10, v65, v59
	v_lshlrev_b64 v[48:49], 1, v[48:49]
	v_add_co_ci_u32_e64 v66, null, v66, v57, s10
	v_add_co_u32 v91, s10, v91, v59
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v92, null, v92, v57, s10
	v_add_co_u32 v93, s10, v93, v59
	v_add_co_ci_u32_e64 v94, null, v94, v57, s10
	v_lshlrev_b64 v[111:112], 1, v[65:66]
	v_add_co_u32 v65, s10, s18, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v66, null, s19, v49, s10
	v_lshlrev_b64 v[113:114], 1, v[91:92]
	v_lshlrev_b64 v[115:116], 1, v[93:94]
	v_add_co_u32 v48, s10, s20, v48
	global_load_b128 v[91:94], v[65:66], off
	v_add_co_ci_u32_e64 v49, null, s21, v49, s10
	v_add_co_u32 v65, s10, s18, v111
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v66, null, s19, v112, s10
	v_add_co_u32 v103, s10, s18, v113
	v_add_co_ci_u32_e64 v104, null, s19, v114, s10
	v_add_co_u32 v107, s10, s18, v115
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v108, null, s19, v116, s10
	global_load_b128 v[95:98], v[48:49], off
	s_clause 0x2
	global_load_b128 v[99:102], v[65:66], off
	global_load_b128 v[103:106], v[103:104], off
	global_load_b128 v[107:110], v[107:108], off
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v85, 0
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v87, v67
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v80, 0
	v_cmp_gt_i32_e64 s9, s30, v79
	v_cmp_gt_i32_e32 vcc_lo, s35, v53
	v_cmp_gt_i32_e64 s0, s36, v53
	v_cmp_ge_i32_e64 s11, s36, v53
	v_cmp_ge_i32_e64 s6, s35, v72
	v_cmp_ge_i32_e64 s4, s36, v72
	s_and_b32 vcc_lo, s9, vcc_lo
	s_and_b32 s0, s9, s0
	v_cmp_ge_i32_e64 s2, s37, v72
	v_cmp_gt_i32_e64 s15, s30, v72
	v_cmp_ge_i32_e64 s10, s35, v53
	v_cmp_ge_i32_e64 s8, s35, v68
	v_cmp_ge_i32_e64 s5, s36, v68
	v_cmp_ge_i32_e64 s3, s37, v68
	v_cmp_gt_i32_e64 s14, s30, v68
	v_cmp_ge_i32_e64 s12, s37, v53
	v_cmp_gt_i32_e64 s1, s37, v53
	v_cmp_gt_i32_e64 s7, s31, v53
	v_cmp_gt_i32_e64 s13, s30, v53
	s_and_b32 s1, s9, s1
	s_and_b32 s7, s9, s7
	s_waitcnt vmcnt(4)
	v_dot2acc_f32_f16 v69, v16, v91
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_dot2acc_f32_f16 v70, v20, v91 :: v_dual_dot2acc_f32_f16 v69, v17, v92
	v_dual_dot2acc_f32_f16 v71, v24, v91 :: v_dual_dot2acc_f32_f16 v70, v21, v92
	v_dot2acc_f32_f16 v73, v28, v91
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2acc_f32_f16 v69, v18, v93
	v_dual_dot2acc_f32_f16 v71, v25, v92 :: v_dual_dot2acc_f32_f16 v70, v22, v93
	s_waitcnt vmcnt(2)
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_dot2acc_f32_f16 v74, v16, v99 :: v_dual_dot2acc_f32_f16 v73, v29, v92
	v_dot2acc_f32_f16 v76, v24, v99
	v_dot2acc_f32_f16 v75, v20, v99
	s_waitcnt vmcnt(1)
	v_dot2acc_f32_f16 v78, v16, v103
	v_dual_dot2acc_f32_f16 v82, v20, v103 :: v_dual_dot2acc_f32_f16 v71, v26, v93
	s_waitcnt vmcnt(0)
	v_dual_dot2acc_f32_f16 v83, v16, v107 :: v_dual_dot2acc_f32_f16 v74, v17, v100
	v_dual_dot2acc_f32_f16 v84, v20, v107 :: v_dual_dot2acc_f32_f16 v75, v21, v100
	v_dual_dot2acc_f32_f16 v77, v28, v99 :: v_dual_dot2acc_f32_f16 v76, v25, v100
	v_dot2acc_f32_f16 v86, v28, v107
	v_dot2acc_f32_f16 v82, v21, v104
	v_dual_dot2acc_f32_f16 v83, v17, v108 :: v_dual_dot2acc_f32_f16 v74, v18, v101
	v_dual_dot2acc_f32_f16 v84, v21, v108 :: v_dual_dot2acc_f32_f16 v69, v19, v94
	v_dot2acc_f32_f16 v77, v29, v100
	v_dual_dot2acc_f32_f16 v86, v29, v108 :: v_dual_dot2acc_f32_f16 v71, v27, v94
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_dot2acc_f32_f16 v75, v22, v101 :: v_dual_dot2acc_f32_f16 v74, v19, v102
	ds_bpermute_b32 v48, v60, v69
	v_dual_dot2acc_f32_f16 v80, v28, v103 :: v_dual_dot2acc_f32_f16 v73, v30, v93
	v_dot2acc_f32_f16 v85, v24, v107
	ds_bpermute_b32 v65, v60, v71
	ds_bpermute_b32 v67, v60, v74
	v_dot2acc_f32_f16 v80, v29, v104
	v_dual_dot2acc_f32_f16 v76, v26, v101 :: v_dual_dot2acc_f32_f16 v85, v25, v108
	v_dot2acc_f32_f16 v77, v30, v101
	v_dual_dot2acc_f32_f16 v82, v22, v105 :: v_dual_dot2acc_f32_f16 v75, v23, v102
	v_dual_dot2acc_f32_f16 v81, v24, v103 :: v_dual_dot2acc_f32_f16 v78, v17, v104
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_dot2acc_f32_f16 v83, v18, v109 :: v_dual_dot2acc_f32_f16 v82, v23, v106
	v_dot2acc_f32_f16 v86, v30, v109
	v_dot2acc_f32_f16 v76, v27, v102
	v_dual_dot2acc_f32_f16 v80, v30, v105 :: v_dual_dot2acc_f32_f16 v77, v31, v102
	v_dot2acc_f32_f16 v85, v26, v109
	v_dual_dot2acc_f32_f16 v81, v25, v104 :: v_dual_dot2acc_f32_f16 v70, v23, v94
	v_dual_dot2acc_f32_f16 v73, v31, v94 :: v_dual_dot2acc_f32_f16 v78, v18, v105
	s_delay_alu instid0(VALU_DEP_4)
	v_dot2acc_f32_f16 v80, v31, v106
	v_dot2acc_f32_f16 v83, v19, v110
	s_waitcnt lgkmcnt(2)
	v_dual_dot2acc_f32_f16 v85, v27, v110 :: v_dual_add_f32 v48, v69, v48
	s_waitcnt lgkmcnt(0)
	v_dual_dot2acc_f32_f16 v86, v31, v110 :: v_dual_add_f32 v67, v74, v67
	ds_bpermute_b32 v49, v60, v70
	ds_bpermute_b32 v66, v60, v73
	ds_bpermute_b32 v91, v60, v83
	ds_bpermute_b32 v92, v60, v75
	ds_bpermute_b32 v102, v60, v86
	v_add_f32_e32 v65, v71, v65
	ds_bpermute_b32 v71, v61, v48
	ds_bpermute_b32 v93, v60, v82
	v_dual_dot2acc_f32_f16 v81, v26, v105 :: v_dual_dot2acc_f32_f16 v78, v19, v106
	v_dot2acc_f32_f16 v84, v22, v109
	ds_bpermute_b32 v99, v60, v76
	ds_bpermute_b32 v101, v60, v85
	v_dot2acc_f32_f16 v81, v27, v106
	ds_bpermute_b32 v79, v60, v78
	v_dot2acc_f32_f16 v84, v23, v110
	ds_bpermute_b32 v104, v60, v77
	ds_bpermute_b32 v103, v60, v80
	ds_bpermute_b32 v100, v60, v81
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v49, v70, v49
	ds_bpermute_b32 v94, v60, v84
	s_waitcnt lgkmcnt(12)
	v_add_f32_e32 v66, v73, v66
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v70, v83, v91
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v73, v75, v92
	s_waitcnt lgkmcnt(8)
	v_dual_add_f32 v83, v86, v102 :: v_dual_add_f32 v48, v48, v71
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v74, v82, v93
	ds_bpermute_b32 v82, v61, v65
	ds_bpermute_b32 v92, v61, v73
	ds_bpermute_b32 v102, v61, v83
	ds_bpermute_b32 v71, v62, v48
	ds_bpermute_b32 v93, v61, v74
	s_waitcnt lgkmcnt(11)
	v_add_f32_e32 v76, v76, v99
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v80, v80, v103
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v75, v84, v94
	ds_bpermute_b32 v84, v61, v66
	ds_bpermute_b32 v91, v61, v70
	v_add_f32_e32 v77, v77, v104
	ds_bpermute_b32 v103, v61, v80
	ds_bpermute_b32 v94, v61, v75
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v65, v65, v82
	s_waitcnt lgkmcnt(5)
	v_dual_add_f32 v73, v73, v92 :: v_dual_add_f32 v48, v48, v71
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v74, v74, v93
	ds_bpermute_b32 v82, v62, v65
	ds_bpermute_b32 v92, v62, v73
	ds_bpermute_b32 v71, v63, v48
	ds_bpermute_b32 v93, v62, v74
	s_waitcnt lgkmcnt(7)
	v_dual_add_f32 v83, v83, v102 :: v_dual_add_f32 v66, v66, v84
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v70, v70, v91
	ds_bpermute_b32 v104, v61, v77
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v80, v80, v103
	ds_bpermute_b32 v84, v62, v66
	ds_bpermute_b32 v91, v62, v70
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v75, v75, v94
	ds_bpermute_b32 v103, v62, v80
	s_waitcnt lgkmcnt(5)
	v_dual_add_f32 v65, v65, v82 :: v_dual_add_f32 v48, v48, v71
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v74, v74, v93
	ds_bpermute_b32 v102, v62, v83
	ds_bpermute_b32 v82, v63, v65
	ds_bpermute_b32 v71, v64, v48
	ds_bpermute_b32 v93, v63, v74
	s_waitcnt lgkmcnt(7)
	v_add_f32_e32 v77, v77, v104
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v66, v66, v84
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v70, v70, v91
	ds_bpermute_b32 v94, v62, v75
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v80, v80, v103
	ds_bpermute_b32 v104, v62, v77
	ds_bpermute_b32 v84, v63, v66
	ds_bpermute_b32 v91, v63, v70
	ds_bpermute_b32 v103, v63, v80
	s_waitcnt lgkmcnt(6)
	v_dual_add_f32 v83, v83, v102 :: v_dual_add_f32 v48, v48, v71
	v_add_f32_e32 v69, v78, v79
	ds_bpermute_b32 v78, v61, v49
	v_add_f32_e32 v79, v81, v100
	v_add_f32_e32 v81, v85, v101
	ds_bpermute_b32 v85, v61, v67
	ds_bpermute_b32 v99, v61, v76
	v_mul_f32_e32 v48, s16, v48
	ds_bpermute_b32 v100, v61, v79
	ds_bpermute_b32 v86, v61, v69
	ds_bpermute_b32 v101, v61, v81
	ds_bpermute_b32 v102, v63, v83
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v70, v70, v91
	v_add_f32_e32 v74, v74, v93
	v_cndmask_b32_e64 v48, 0xff800000, v48, s10
	s_waitcnt lgkmcnt(7)
	v_dual_add_f32 v75, v75, v94 :: v_dual_add_f32 v80, v80, v103
	v_add_f32_e32 v66, v66, v84
	ds_bpermute_b32 v91, v64, v70
	ds_bpermute_b32 v93, v64, v74
	ds_bpermute_b32 v94, v63, v75
	ds_bpermute_b32 v103, v64, v80
	s_waitcnt lgkmcnt(10)
	v_add_f32_e32 v49, v49, v78
	ds_bpermute_b32 v84, v64, v66
	s_waitcnt lgkmcnt(9)
	v_dual_add_f32 v67, v67, v85 :: v_dual_add_f32 v76, v76, v99
	ds_bpermute_b32 v78, v62, v49
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v79, v79, v100
	s_waitcnt lgkmcnt(8)
	v_add_f32_e32 v69, v69, v86
	ds_bpermute_b32 v85, v62, v67
	ds_bpermute_b32 v99, v62, v76
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v81, v81, v101
	ds_bpermute_b32 v100, v62, v79
	ds_bpermute_b32 v86, v62, v69
	s_waitcnt lgkmcnt(9)
	v_add_f32_e32 v70, v70, v91
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v66, v66, v84
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v70, s16, v70
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v49, v49, v78
	s_waitcnt lgkmcnt(3)
	v_dual_mul_f32 v66, s16, v66 :: v_dual_add_f32 v67, v67, v85
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v76, v76, v99
	ds_bpermute_b32 v101, v62, v81
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v79, v79, v100
	s_waitcnt lgkmcnt(1)
	v_add_f32_e32 v69, v69, v86
	ds_bpermute_b32 v85, v63, v67
	ds_bpermute_b32 v99, v63, v76
	v_add_f32_e32 v73, v73, v92
	ds_bpermute_b32 v100, v63, v79
	ds_bpermute_b32 v78, v63, v49
	ds_bpermute_b32 v86, v63, v69
	s_waitcnt lgkmcnt(5)
	v_add_f32_e32 v81, v81, v101
	s_waitcnt lgkmcnt(3)
	v_dual_add_f32 v67, v67, v85 :: v_dual_add_f32 v76, v76, v99
	ds_bpermute_b32 v101, v63, v81
	ds_bpermute_b32 v92, v63, v73
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v79, v79, v100
	ds_bpermute_b32 v85, v64, v67
	ds_bpermute_b32 v99, v64, v76
	ds_bpermute_b32 v100, v64, v79
	s_waitcnt lgkmcnt(6)
	v_add_f32_e32 v49, v49, v78
	s_waitcnt lgkmcnt(4)
	v_add_f32_e32 v81, v81, v101
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v67, v67, v85
	ds_bpermute_b32 v101, v64, v81
	v_add_f32_e32 v75, v75, v94
	v_mul_f32_e32 v67, s16, v67
	v_add_f32_e32 v83, v83, v102
	ds_bpermute_b32 v94, v64, v75
	v_add_f32_e32 v77, v77, v104
	ds_bpermute_b32 v102, v64, v83
	v_add_f32_e32 v73, v73, v92
	ds_bpermute_b32 v92, v64, v73
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v71, v73, v92
	v_dual_add_f32 v73, v74, v93 :: v_dual_add_f32 v74, v75, v94
	v_dual_add_f32 v75, v76, v99 :: v_dual_add_f32 v76, v79, v100
	ds_bpermute_b32 v78, v64, v49
	v_add_f32_e32 v69, v69, v86
	v_mul_f32_e32 v71, s16, v71
	v_dual_add_f32 v79, v80, v103 :: v_dual_add_f32 v80, v83, v102
	v_add_f32_e32 v65, v65, v82
	ds_bpermute_b32 v104, v63, v77
	v_cndmask_b32_e64 v84, 0xff800000, v71, s0
	v_mul_f32_e32 v68, s40, v80
	ds_bpermute_b32 v82, v64, v65
	v_cndmask_b32_e32 v80, 0xff800000, v67, vcc_lo
	s_and_b32 vcc_lo, s14, s8
	s_waitcnt lgkmcnt(2)
	v_add_f32_e32 v49, v49, v78
	v_add_f32_e32 v78, v81, v101
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v49, s16, v49
	v_cndmask_b32_e64 v72, 0xff800000, v49, s11
	v_mul_f32_e32 v49, s16, v73
	v_dual_mul_f32 v73, s16, v74 :: v_dual_mul_f32 v74, s16, v75
	ds_bpermute_b32 v86, v64, v69
	v_cndmask_b32_e64 v74, 0xff800000, v74, s1
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v69, v69, v86
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v69, s16, v69
	v_cndmask_b32_e32 v81, 0xff800000, v69, vcc_lo
	s_and_b32 vcc_lo, s15, s6
	v_cndmask_b32_e32 v83, 0xff800000, v70, vcc_lo
	s_and_b32 vcc_lo, s14, s5
	v_cndmask_b32_e32 v85, 0xff800000, v49, vcc_lo
	v_max3_f32 v49, v87, v48, v80
	s_and_b32 vcc_lo, s15, s4
	v_cndmask_b32_e32 v73, 0xff800000, v73, vcc_lo
	s_and_b32 vcc_lo, s14, s3
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_max3_f32 v67, v49, v81, v83
	v_add_f32_e32 v65, v65, v82
	v_cndmask_b32_e64 v82, 0xff800000, v68, s15
	v_dual_sub_f32 v48, v48, v67 :: v_dual_mul_f32 v65, s16, v65
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v48, v48
	v_cndmask_b32_e64 v75, 0xff800000, v65, s12
	v_mul_f32_e32 v65, s16, v76
	v_dual_add_f32 v77, v77, v104 :: v_dual_mul_f32 v76, s16, v78
	v_cndmask_b32_e64 v78, 0xff800000, v66, s13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_max3_f32 v68, v88, v75, v74
	v_cndmask_b32_e32 v86, 0xff800000, v65, vcc_lo
	ds_bpermute_b32 v104, v64, v77
	s_and_b32 vcc_lo, s15, s2
	v_cmp_eq_f32_e64 s2, 0xff800000, v67
	v_cndmask_b32_e32 v76, 0xff800000, v76, vcc_lo
	v_max3_f32 v65, v89, v72, v84
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v48, v48, 0, s2
	v_max3_f32 v66, v65, v85, v73
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_max3_f32 v65, v68, v86, v76
	v_sub_f32_e32 v68, v87, v67
	v_fma_mix_f32 v92, v95, v48, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v100, v97, v48, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mul_f32 v79, s39, v79 :: v_dual_sub_f32 v70, v75, v65
	v_exp_f32_e32 v68, v68
	v_cmp_eq_f32_e64 s0, 0xff800000, v65
	v_cmp_eq_f32_e64 s1, 0xff800000, v66
	s_delay_alu instid0(VALU_DEP_3)
	v_cndmask_b32_e64 v79, 0xff800000, v79, s14
	s_waitcnt lgkmcnt(0)
	v_add_f32_e32 v77, v77, v104
	v_exp_f32_e32 v70, v70
	v_fma_mix_f32 v99, v97, v48, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v91, v95, v48, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v93, v96, v48, neg(0) op_sel_hi:[1,0,0]
	v_mul_f32_e32 v77, s38, v77
	v_fma_mix_f32 v94, v96, v48, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v101, v98, v48, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v102, v98, v48, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_sub_f32_e32 v74, v74, v65
	v_cndmask_b32_e64 v77, 0xff800000, v77, s7
	v_sub_f32_e32 v76, v76, v65
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_exp_f32_e32 v74, v74
	v_max3_f32 v69, v90, v78, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_exp_f32_e32 v76, v76
	v_max3_f32 v49, v69, v79, v82
	v_sub_f32_e32 v69, v72, v66
	v_cndmask_b32_e64 v72, v68, 0, s2
	v_sub_f32_e32 v68, v89, v66
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_f32_e32 v82, v82, v49
	v_exp_f32_e32 v69, v69
	v_sub_f32_e32 v71, v78, v49
	s_delay_alu instid0(VALU_DEP_3)
	v_exp_f32_e32 v68, v68
	v_cndmask_b32_e64 v78, v70, 0, s0
	v_fmac_f32_e32 v100, v72, v41
	v_fmac_f32_e32 v92, v72, v45
	v_exp_f32_e32 v71, v71
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v49
	v_fma_mix_f32 v120, v96, v78, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v99, v72, v40
	v_cndmask_b32_e64 v75, v69, 0, s1
	v_add_co_u32 v40, s3, s20, v111
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v41, null, s21, v112, s3
	v_fma_mix_f32 v108, v97, v75, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v89, v68, 0, s1
	v_fmac_f32_e32 v91, v72, v44
	v_add_co_u32 v44, s3, s20, v113
	v_fma_mix_f32 v118, v95, v78, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v93, v72, v46
	v_add_co_ci_u32_e64 v45, null, s21, v114, s3
	v_add_co_u32 v46, s3, s20, v115
	v_fma_mix_f32 v103, v95, v75, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v104, v95, v75, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v105, v96, v75, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v106, v96, v75, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v94, v72, v47
	v_add_co_ci_u32_e64 v47, null, s21, v116, s3
	v_cndmask_b32_e64 v87, v71, 0, vcc_lo
	v_fma_mix_f32 v124, v98, v78, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fmac_f32_e32 v103, v89, v36
	v_fmac_f32_e32 v104, v89, v37
	v_fmac_f32_e32 v105, v89, v38
	v_fmac_f32_e32 v106, v89, v39
	s_clause 0x1
	global_load_b128 v[36:39], v[44:45], off
	global_load_b128 v[68:71], v[46:47], off
	v_dual_mov_b32 v46, v55 :: v_dual_sub_f32 v55, v90, v49
	v_fma_mix_f32 v119, v96, v78, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v126, v96, v87, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v96, v96, v87, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v128, v98, v87, neg(0) op_sel_hi:[1,0,0]
	v_exp_f32_e32 v55, v55
	v_dual_mov_b32 v44, v56 :: v_dual_mov_b32 v45, v58
	v_sub_f32_e32 v56, v80, v67
	v_sub_f32_e32 v58, v84, v66
	v_sub_f32_e32 v80, v81, v67
	v_dual_sub_f32 v84, v86, v65 :: v_dual_sub_f32 v81, v83, v67
	v_sub_f32_e32 v83, v85, v66
	v_fma_mix_f32 v122, v97, v78, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	s_delay_alu instid0(TRANS32_DEP_1)
	v_cndmask_b32_e64 v85, v55, 0, vcc_lo
	v_dual_mov_b32 v47, v54 :: v_dual_sub_f32 v54, v88, v65
	v_fmac_f32_e32 v102, v72, v43
	v_exp_f32_e32 v56, v56
	v_exp_f32_e32 v58, v58
	v_fma_mix_f32 v117, v95, v78, neg(0) op_sel_hi:[1,0,0]
	v_exp_f32_e32 v54, v54
	v_fma_mix_f32 v121, v97, v78, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v123, v98, v78, neg(0) op_sel_hi:[1,0,0]
	v_exp_f32_e32 v80, v80
	v_fma_mix_f32 v127, v97, v87, neg(0) op_sel_hi:[1,0,0]
	v_dual_fmac_f32 v96, v85, v7 :: v_dual_sub_f32 v79, v79, v49
	v_cndmask_b32_e64 v55, v56, 0, s2
	s_delay_alu instid0(TRANS32_DEP_3) | instskip(SKIP_1) | instid1(TRANS32_DEP_2)
	v_cndmask_b32_e64 v56, v58, 0, s1
	v_cndmask_b32_e64 v58, v74, 0, s0
	v_cndmask_b32_e64 v54, v54, 0, s0
	v_fmac_f32_e32 v101, v72, v42
	global_load_b128 v[40:43], v[40:41], off
	v_add_f32_e32 v48, v48, v55
	v_dual_add_f32 v78, v78, v58 :: v_dual_sub_f32 v77, v77, v49
	v_exp_f32_e32 v83, v83
	v_exp_f32_e32 v84, v84
	v_exp_f32_e32 v79, v79
	v_fmac_f32_e32 v127, v85, v0
	v_exp_f32_e32 v77, v77
	v_exp_f32_e32 v82, v82
	v_fma_mix_f32 v109, v98, v75, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v110, v98, v75, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v98, v98, v87, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v107, v97, v75, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v125, v95, v87, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v95, v95, v87, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v79, v79, 0, vcc_lo
	v_fma_mix_f32 v97, v97, v87, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_cndmask_b32_e64 v74, v77, 0, vcc_lo
	v_cndmask_b32_e64 v77, v80, 0, s2
	v_dual_fmac_f32 v98, v85, v3 :: v_dual_add_nc_u32 v53, 32, v53
	v_cndmask_b32_e64 v76, v76, 0, s0
	v_cndmask_b32_e64 v82, v82, 0, vcc_lo
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_e32 v0, v48, v77
	v_exp_f32_e32 v81, v81
	v_add_f32_e32 v75, v75, v56
	v_fmac_f32_e32 v117, v54, v12
	v_fmac_f32_e32 v118, v54, v13
	v_fmac_f32_e32 v119, v54, v14
	v_fmac_f32_e32 v120, v54, v15
	v_fmac_f32_e32 v121, v54, v8
	v_fmac_f32_e32 v122, v54, v9
	v_fmac_f32_e32 v123, v54, v10
	v_fmac_f32_e32 v124, v54, v11
	v_cndmask_b32_e64 v80, v81, 0, s2
	v_cndmask_b32_e64 v81, v83, 0, s1
	v_cndmask_b32_e64 v83, v84, 0, s0
	v_add_f32_e32 v84, v87, v74
	v_fmac_f32_e32 v128, v85, v2
	v_fmac_f32_e32 v108, v89, v33
	v_fmac_f32_e32 v110, v89, v35
	v_add_f32_e32 v2, v78, v83
	v_add_f32_e32 v3, v84, v79
	v_fmac_f32_e32 v107, v89, v32
	v_fmac_f32_e32 v125, v85, v4
	v_fmac_f32_e32 v95, v85, v5
	v_fmac_f32_e32 v126, v85, v6
	v_fmac_f32_e32 v97, v85, v1
	v_add_f32_e32 v1, v75, v81
	v_cmp_le_i32_e64 s3, s30, v53
	s_or_b32 s17, s3, s17
	s_waitcnt vmcnt(0)
	v_fma_mix_f32 v4, v55, v40, v91 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v5, v55, v40, v92 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v6, v55, v41, v93 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v7, v55, v41, v94 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v8, v55, v42, v99 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v55, v42, v100 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v10, v55, v43, v101 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v11, v55, v43, v102 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v48, v58, v40, v117 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v75, v58, v40, v118 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v78, v58, v41, v119 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v84, v58, v41, v120 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v86, v58, v42, v121 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v87, v58, v42, v122 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v88, v58, v43, v123 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v90, v58, v43, v124 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_add_f32_e32 v58, v0, v80
	v_dual_add_f32 v55, v2, v76 :: v_dual_add_f32 v0, v3, v82
	v_sub_f32_e32 v73, v73, v66
	v_fma_mix_f32 v12, v56, v40, v103 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v13, v56, v40, v104 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_fmac_f32 v55, v46, v54 :: v_dual_mov_b32 v54, v0
	v_exp_f32_e32 v73, v73
	v_fmac_f32_e32 v109, v89, v34
	v_fma_mix_f32 v14, v56, v41, v105 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v15, v56, v41, v106 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v32, v56, v42, v107 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v33, v56, v42, v108 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v35, v56, v43, v110 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v91, v74, v40, v125 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v40, v74, v40, v95 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v92, v74, v41, v126 op_sel_hi:[0,1,0]
	v_cndmask_b32_e64 v73, v73, 0, s1
	v_fma_mix_f32 v41, v74, v41, v96 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v93, v74, v42, v127 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v42, v74, v42, v97 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v94, v74, v43, v128 op_sel_hi:[0,1,0]
	v_fmac_f32_e32 v54, v47, v85
	v_fma_mix_f32 v34, v56, v43, v109 op_sel_hi:[0,1,0]
	v_add_f32_e32 v56, v1, v73
	v_fma_mix_f32 v1, v74, v43, v98 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fmac_f32_e32 v58, v45, v72
	v_fma_mix_f32 v0, v77, v36, v4 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v2, v77, v36, v5 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fmac_f32_e32 v56, v44, v89
	v_fma_mix_f32 v3, v77, v37, v6 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v77, v37, v7 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v5, v77, v38, v8 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v6, v77, v38, v9 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v7, v77, v39, v10 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v8, v77, v39, v11 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v81, v36, v12 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v10, v81, v36, v13 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v11, v81, v37, v14 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v12, v81, v37, v15 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v13, v81, v38, v32 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v14, v81, v38, v33 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v15, v81, v39, v34 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v35, v81, v39, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v48, v83, v36, v48 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v72, v83, v36, v75 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v74, v83, v37, v78 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v75, v83, v37, v84 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v77, v83, v38, v86 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v78, v83, v38, v87 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v81, v83, v39, v88 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v83, v83, v39, v90 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v84, v79, v36, v91 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v86, v79, v36, v40 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v87, v79, v37, v92 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v88, v79, v37, v41 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v89, v79, v38, v93 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v90, v79, v38, v42 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v91, v79, v39, v94 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v79, v79, v39, v1 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v44, v80, v68, v0 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v45, v80, v68, v2 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v46, v80, v69, v3 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v47, v80, v69, v4 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v40, v80, v70, v5 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v41, v80, v70, v6 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v42, v80, v71, v7 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v43, v80, v71, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v36, v73, v68, v9 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v37, v73, v68, v10 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v38, v73, v69, v11 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v39, v73, v69, v12 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v32, v73, v70, v13 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v33, v73, v70, v14 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v34, v73, v71, v15 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v35, v73, v71, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v12, v76, v68, v48 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v13, v76, v68, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v14, v76, v69, v74 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v15, v76, v69, v75 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v8, v76, v70, v77 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v9, v76, v70, v78 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v10, v76, v71, v81 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v11, v76, v71, v83 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v4, v82, v68, v84 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v5, v82, v68, v86 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v6, v82, v69, v87 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v7, v82, v69, v88 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v0, v82, v70, v89 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v1, v82, v70, v90 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	v_fma_mix_f32 v2, v82, v71, v91 op_sel_hi:[0,1,0]
	v_fma_mix_f32 v3, v82, v71, v79 op_sel:[0,1,0] op_sel_hi:[0,1,0]
	s_and_not1_b32 exec_lo, exec_lo, s17
	s_cbranch_execnz .LBB0_2
; %bb.3:
	s_or_b32 exec_lo, exec_lo, s17
.LBB0_4:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s34
	s_lshl_b64 s[0:1], s[28:29], 3
	s_ashr_i32 s2, s33, 31
	s_add_u32 s0, s0, s33
	s_addc_u32 s1, s1, s2
	v_add_co_u32 v16, s0, s0, v51
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_ci_u32_e64 v17, null, s1, 0, s0
	v_lshlrev_b32_e32 v20, 2, v52
	v_lshlrev_b64 v[18:19], 2, v[16:17]
	v_lshlrev_b64 v[16:17], 12, v[16:17]
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_co_u32 v20, s0, s22, v20
	v_add_co_ci_u32_e64 v21, null, s23, 0, s0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64 v[18:19], 2, v[18:19]
	v_add_co_u32 v16, vcc_lo, v20, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v17, null, v21, v17, vcc_lo
	v_cmp_eq_u32_e32 vcc_lo, 0, v50
	s_clause 0x1
	global_store_b128 v[16:17], v[44:47], off
	global_store_b128 v[16:17], v[40:43], off offset:16
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_6
; %bb.5:
	v_add_co_u32 v20, s0, s24, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v21, null, s25, v19, s0
	v_add_co_u32 v22, s0, s26, v18
	v_add_co_ci_u32_e64 v23, null, s27, v19, s0
	global_store_b32 v[20:21], v67, off
	global_store_b32 v[22:23], v58, off
.LBB0_6:
	s_or_b32 exec_lo, exec_lo, s1
	s_clause 0x1
	global_store_b128 v[16:17], v[36:39], off offset:1024
	global_store_b128 v[16:17], v[32:35], off offset:1040
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_8
; %bb.7:
	v_add_co_u32 v20, s0, s24, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v21, null, s25, v19, s0
	v_add_co_u32 v22, s0, s26, v18
	v_add_co_ci_u32_e64 v23, null, s27, v19, s0
	global_store_b32 v[20:21], v66, off offset:4
	global_store_b32 v[22:23], v56, off offset:4
.LBB0_8:
	s_or_b32 exec_lo, exec_lo, s1
	s_clause 0x1
	global_store_b128 v[16:17], v[12:15], off offset:2048
	global_store_b128 v[16:17], v[8:11], off offset:2064
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_10
; %bb.9:
	v_add_co_u32 v8, s0, s24, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v9, null, s25, v19, s0
	v_add_co_u32 v10, s0, s26, v18
	v_add_co_ci_u32_e64 v11, null, s27, v19, s0
	global_store_b32 v[8:9], v65, off offset:8
	global_store_b32 v[10:11], v55, off offset:8
.LBB0_10:
	s_or_b32 exec_lo, exec_lo, s1
	s_clause 0x1
	global_store_b128 v[16:17], v[4:7], off offset:3072
	global_store_b128 v[16:17], v[0:3], off offset:3088
	s_and_saveexec_b32 s0, vcc_lo
	s_cbranch_execz .LBB0_12
; %bb.11:
	v_add_co_u32 v0, vcc_lo, s24, v18
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s25, v19, vcc_lo
	v_add_co_u32 v2, vcc_lo, s26, v18
	v_add_co_ci_u32_e64 v3, null, s27, v19, vcc_lo
	global_store_b32 v[0:1], v49, off offset:12
	global_store_b32 v[2:3], v54, off offset:12
.LBB0_12:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
.Lfunc_end0:
	.size	_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if, .Lfunc_end0-_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
		.amdhsa_group_segment_fixed_size 0
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
		.amdhsa_next_free_vgpr 129
		.amdhsa_next_free_sgpr 41
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
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.num_vgpr, 129
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.num_agpr, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.numbered_sgpr, 41
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.num_named_barrier, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.private_seg_size, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.uses_vcc, 1
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.uses_flat_scratch, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.has_dyn_sized_stack, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.has_recursion, 0
	.set .L_Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5288
; TotalNumSgprs: 43
; NumVgprs: 129
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 16
; NumSGPRsForWavesPerEU: 43
; NumVGPRsForWavesPerEU: 129
; Occupancy: 10
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
	s_lshl_b64 s[0:1], s[12:13], 7
	v_dual_mov_b32 v3, 0 :: v_dual_mov_b32 v4, 0
	s_waitcnt lgkmcnt(0)
	s_add_u32 s6, s6, s0
	s_addc_u32 s7, s7, s1
	s_lshl_b64 s[0:1], s[2:3], 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_u32 s6, s6, s0
	s_addc_u32 s7, s7, s1
	s_clause 0x7
	s_load_b32 s20, s[6:7], 0x10
	s_load_b32 s0, s[6:7], 0x0
	s_load_b32 s19, s[6:7], 0x20
	s_load_b32 s18, s[6:7], 0x30
	s_load_b32 s17, s[6:7], 0x40
	s_load_b32 s16, s[6:7], 0x50
	s_load_b32 s15, s[6:7], 0x60
	s_load_b32 s6, s[6:7], 0x70
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v1, s20
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_f32 v1, s0, 0xff800000, v1
	v_max3_f32 v1, v1, s19, s18
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_max3_f32 v1, v1, s17, s16
	v_max3_f32 v1, v1, s15, s6
	s_delay_alu instid0(VALU_DEP_1)
	v_readfirstlane_b32 s7, v1
	v_cmp_eq_f32_e32 vcc_lo, 0xff800000, v1
	s_cmp_neq_f32 s7, 0xff800000
	s_cselect_b32 s14, -1, 0
	s_cbranch_vccnz .LBB1_2
; %bb.1:
	s_sub_f32 s0, s0, s7
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v4, s0
.LBB1_2:
	v_lshlrev_b32_e32 v2, 2, v0
	s_lshl_b64 s[0:1], s[12:13], 5
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, s3
	v_add_co_u32 v0, s4, s4, v2
	s_lshl_b64 s[22:23], s[0:1], 2
	v_add_co_ci_u32_e64 v1, null, s5, 0, s4
	s_lshl_b64 s[4:5], s[0:1], 10
	s_add_u32 s0, s8, s22
	s_addc_u32 s1, s9, s23
	s_and_b32 s8, s14, exec_lo
	s_cselect_b32 s8, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s8, 1
	s_cbranch_scc1 .LBB1_4
; %bb.3:
	s_sub_f32 s8, s20, s7
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v3, s8
.LBB1_4:
	v_add_co_u32 v0, vcc_lo, v0, s4
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)
	v_add_co_ci_u32_e64 v1, null, s5, v1, vcc_lo
	s_and_b32 s4, s14, exec_lo
	v_dual_mov_b32 v5, 0 :: v_dual_mov_b32 v6, 0
	s_cselect_b32 s4, 1, 0
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_6
; %bb.5:
	s_sub_f32 s4, s19, s7
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v6, s4
.LBB1_6:
	s_and_b32 s4, s14, exec_lo
	s_cselect_b32 s4, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_8
; %bb.7:
	s_sub_f32 s4, s18, s7
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v5, s4
.LBB1_8:
	s_and_b32 s4, s14, exec_lo
	v_dual_mov_b32 v7, 0 :: v_dual_mov_b32 v8, 0
	s_cselect_b32 s4, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_10
; %bb.9:
	s_sub_f32 s4, s17, s7
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v8, s4
.LBB1_10:
	s_and_b32 s4, s14, exec_lo
	s_cselect_b32 s4, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_12
; %bb.11:
	s_sub_f32 s4, s16, s7
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v7, s4
.LBB1_12:
	s_and_b32 s4, s14, exec_lo
	v_dual_mov_b32 v9, 0 :: v_dual_mov_b32 v10, 0
	s_cselect_b32 s4, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB1_14
; %bb.13:
	s_sub_f32 s4, s15, s7
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v10, s4
.LBB1_14:
	v_add_co_u32 v12, vcc_lo, v0, 0x2000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_co_ci_u32_e64 v13, null, 0, v1, vcc_lo
	v_add_co_u32 v14, vcc_lo, v0, 0x4000
	v_add_co_ci_u32_e64 v15, null, 0, v1, vcc_lo
	v_add_co_u32 v18, vcc_lo, v0, 0x6000
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v19, null, 0, v1, vcc_lo
	s_clause 0x6
	global_load_b32 v17, v[12:13], off offset:-4096
	global_load_b32 v16, v[12:13], off
	global_load_b32 v13, v[14:15], off offset:-4096
	global_load_b32 v12, v[14:15], off
	global_load_b32 v14, v[18:19], off offset:-4096
	global_load_b32 v15, v[18:19], off
	global_load_b32 v11, v[0:1], off
	s_clause 0x6
	s_load_b32 s17, s[0:1], 0x0
	s_load_b32 s16, s[0:1], 0x10
	s_load_b32 s15, s[0:1], 0x20
	s_load_b32 s9, s[0:1], 0x30
	s_load_b32 s8, s[0:1], 0x40
	s_load_b32 s5, s[0:1], 0x50
	s_load_b32 s4, s[0:1], 0x60
	s_and_b32 s14, s14, exec_lo
	s_cselect_b32 s14, 1, 0
	s_delay_alu instid0(SALU_CYCLE_1)
	s_cmp_lg_u32 s14, 1
	s_cbranch_scc1 .LBB1_16
; %bb.15:
	s_sub_f32 s6, s6, s7
	s_delay_alu instid0(SALU_CYCLE_3)
	v_exp_f32_e32 v9, s6
.LBB1_16:
	v_add_co_u32 v0, vcc_lo, 0x7000, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_load_b32 s0, s[0:1], 0x70
	global_load_b32 v0, v[0:1], off
	s_waitcnt lgkmcnt(0)
	v_fma_f32 v1, v4, s17, 0
	s_waitcnt vmcnt(1)
	v_fma_f32 v4, v4, v11, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v1, s16, v3 :: v_dual_fmac_f32 v4, v3, v17
	v_dual_fmac_f32 v1, s15, v6 :: v_dual_fmac_f32 v4, v6, v16
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v4, v5, v13
	v_dual_fmac_f32 v1, s9, v5 :: v_dual_fmac_f32 v4, v8, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v1, s8, v8 :: v_dual_fmac_f32 v4, v7, v14
	v_fmac_f32_e32 v1, s5, v7
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_fmac_f32 v4, v10, v15 :: v_dual_fmac_f32 v1, s4, v10
	v_fmac_f32_e32 v1, s0, v9
	s_lshl_b64 s[0:1], s[2:3], 15
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	s_add_u32 s2, s10, s0
	s_addc_u32 s3, s11, s1
	s_lshl_b64 s[0:1], s[12:13], 10
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, s1
	s_waitcnt vmcnt(0)
	v_fmac_f32_e32 v4, v9, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_div_scale_f32 v0, null, v1, v1, v4
	v_div_scale_f32 v6, vcc_lo, v4, v1, v4
	v_rcp_f32_e32 v3, v0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_f32 v5, -v0, v3, 1.0
	v_fmac_f32_e32 v3, v5, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mul_f32_e32 v5, v6, v3
	v_fma_f32 v7, -v0, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fmac_f32_e32 v5, v7, v3
	v_fma_f32 v0, -v0, v5, v6
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_div_fmas_f32 v0, v0, v3, v5
	v_div_fixup_f32 v0, v0, v1, v4
	global_store_b32 v2, v0, s[0:1]
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
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 24
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
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.num_vgpr, 20
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.num_agpr, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.numbered_sgpr, 24
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.num_named_barrier, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.private_seg_size, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.uses_vcc, 1
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.uses_flat_scratch, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.has_dyn_sized_stack, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.has_recursion, 0
	.set .L_Z15reduce_segmentsPKfS0_S0_Pf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1000
; TotalNumSgprs: 26
; NumVgprs: 20
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 26
; NumVGPRsForWavesPerEU: 20
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
	.type	__hip_cuid_9d6cf5c644dabb82,@object ; @__hip_cuid_9d6cf5c644dabb82
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_9d6cf5c644dabb82
__hip_cuid_9d6cf5c644dabb82:
	.byte	0                               ; 0x0
	.size	__hip_cuid_9d6cf5c644dabb82, 1

	.ident	"AMD clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 0bace1908348b840e6aa1b4b6e12151dae208158)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_9d6cf5c644dabb82
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
    .private_segment_fixed_size: 0
    .sgpr_count:     43
    .sgpr_spill_count: 0
    .symbol:         _Z11decode_attnPKDF16_S0_S0_PfS1_S1_if.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     129
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
    .sgpr_count:     26
    .sgpr_spill_count: 0
    .symbol:         _Z15reduce_segmentsPKfS0_S0_Pf.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx1151
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
