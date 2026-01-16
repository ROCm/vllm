; Kernel info:
; codeLenInByte = 2060
; TotalNumSgprs: 46
; NumVgprs: 19
; NumAgprs: 0
; TotalNumVgprs: 19
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 5
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 46
; NumVGPRsForWavesPerEU: 19
; AccumOffset: 20
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 4
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,"axG",@progbits,_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,comdat
	.protected	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf ; -- Begin function _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.globl	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.p2align	8
	.type	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,@function
_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf: ; @_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
.Lfunc_begin8:
	.loc	53 63 0 is_stmt 1               ; csrc/cache_kernels_fused.hip:63:0
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:positions <- [DW_OP_LLVM_poisoned] undef
	s_load_dwordx2 s[18:19], s[0:1], 0x20
.Ltmp874:
	.loc	54 258 116 prologue_end         ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dword s28, s[0:1], 0x28
.Ltmp875:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe <- [DW_OP_LLVM_poisoned] undef
	.loc	54 0 116 is_stmt 0              ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:0:116
	s_load_dwordx2 s[4:5], s[0:1], 0x0
.Ltmp876:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:positions <- [DW_OP_LLVM_poisoned] $sgpr4_sgpr5
	s_load_dwordx4 s[12:15], s[0:1], 0x10
.Ltmp877:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	s_load_dwordx4 s[8:11], s[0:1], 0x58
.Ltmp878:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr10_sgpr11
	.loc	54 258 116                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dword s30, s[0:1], 0x50
.Ltmp879:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:num_q_heads <- [DW_OP_LLVM_poisoned] $sgpr30
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe_stride_head <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_quant_scale <- undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	.loc	53 65 29 is_stmt 1              ; csrc/cache_kernels_fused.hip:65:29
	s_mov_b32 s3, 0
.Ltmp880:
	.loc	54 258 116                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s29, s28, 31
.Ltmp881:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:token_idx <- [DW_OP_LLVM_poisoned] undef
	.loc	53 66 23                        ; csrc/cache_kernels_fused.hip:66:23
	s_lshl_b64 s[6:7], s[2:3], 3
	s_add_u32 s4, s4, s6
.Ltmp882:
	s_addc_u32 s5, s5, s7
	s_load_dwordx2 s[20:21], s[4:5], 0x0
.Ltmp883:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:cos_sin_ptr <- undef
	.loc	53 70 33                        ; csrc/cache_kernels_fused.hip:70:33
	s_lshr_b32 s4, s28, 31
	s_add_i32 s4, s28, s4
	s_ashr_i32 s16, s4, 1
.Ltmp884:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	.loc	53 73 30                        ; csrc/cache_kernels_fused.hip:73:30
	s_mul_i32 s30, s30, s16
.Ltmp885:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr30
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	53 74 31                        ; csrc/cache_kernels_fused.hip:74:31
	v_cmp_gt_i32_e32 vcc, s30, v0
.Ltmp886:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe_stride_token <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe_stride <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c_stride <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_size <- [DW_OP_LLVM_poisoned] undef
	.loc	53 74 3 is_stmt 0               ; csrc/cache_kernels_fused.hip:74:3
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB8_3
.Ltmp887:
; %bb.1:                                ; %.lr.ph
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr10_sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr30
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	53 0 3                          ; csrc/cache_kernels_fused.hip:0:3
	s_load_dwordx2 s[26:27], s[0:1], 0x8
.Ltmp888:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	.loc	54 258 116 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dwordx4 s[4:7], s[0:1], 0x30
.Ltmp889:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe_stride_head <- [DW_OP_LLVM_poisoned] $sgpr6_sgpr7
	.loc	53 68 54                        ; csrc/cache_kernels_fused.hip:68:54
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s17, s20, s29
	s_mul_hi_u32 s24, s20, s28
	s_add_i32 s17, s24, s17
	s_mul_i32 s24, s21, s28
	s_add_i32 s25, s17, s24
	s_mul_i32 s24, s20, s28
	.loc	53 68 48 is_stmt 0              ; csrc/cache_kernels_fused.hip:68:48
	s_lshl_b64 s[24:25], s[24:25], 1
	s_add_u32 s24, s18, s24
	s_mul_i32 s5, s5, s2
	s_mul_hi_u32 s31, s4, s2
	s_addc_u32 s25, s19, s25
	s_add_i32 s5, s31, s5
	s_mul_i32 s4, s4, s2
	s_ashr_i32 s17, s16, 31
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s4, s26, s4
	s_addc_u32 s5, s27, s5
	s_abs_i32 s31, s16
	v_cvt_f32_u32_e32 v1, s31
	s_load_dword s26, s[0:1], 0x8c
.Ltmp890:
	.loc	53 0 48                         ; csrc/cache_kernels_fused.hip:0:48
	s_sub_i32 s35, 0, s31
.Ltmp891:
	.loc	53 80 30 is_stmt 1              ; csrc/cache_kernels_fused.hip:80:30
	s_sub_i32 s34, 0, s16
	v_rcp_iflag_f32_e32 v1, v1
	s_waitcnt lgkmcnt(0)
	s_and_b32 s33, s26, 0xffff
	s_mov_b64 s[26:27], 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v2, s35, v1
	v_mul_hi_u32 v2, v1, v2
	v_add_u32_e32 v1, v1, v2
	v_mov_b32_e32 v2, v0
.Ltmp892:
.LBB8_2:                                ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr10_sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe_stride_head <- [DW_OP_LLVM_poisoned] $sgpr6_sgpr7
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr30
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr2
	.loc	53 75 22                        ; csrc/cache_kernels_fused.hip:75:22
	v_sub_u32_e32 v4, 0, v2
	v_max_i32_e32 v4, v2, v4
	v_mul_hi_u32 v5, v4, v1
	v_mul_lo_u32 v6, v5, s31
	v_sub_u32_e32 v4, v4, v6
	v_add_u32_e32 v7, 1, v5
	v_cmp_le_u32_e32 vcc, s31, v4
	v_subrev_u32_e32 v6, s31, v4
	v_xor_b32_e32 v3, s16, v2
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_cndmask_b32_e32 v4, v4, v6, vcc
	v_add_u32_e32 v6, 1, v5
	v_cmp_le_u32_e32 vcc, s31, v4
	v_ashrrev_i32_e32 v3, 31, v3
	s_nop 0
	v_cndmask_b32_e32 v4, v5, v6, vcc
	v_xor_b32_e32 v4, v4, v3
	v_sub_u32_e32 v6, v4, v3
.Ltmp893:
	;DEBUG_VALUE: head_idx <- [DW_OP_LLVM_poisoned] $vgpr6
	;DEBUG_VALUE: pair_idx <- [DW_OP_LLVM_poisoned] $vgpr2, $vgpr6, $sgpr16
	;DEBUG_VALUE: cos <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: sin <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: pair_idx_x <- [DW_OP_LLVM_poisoned] $vgpr2, $vgpr6, $sgpr16
	;DEBUG_VALUE: pair_idx_y <- [DW_OP_LLVM_poisoned] $vgpr2, $sgpr16, $vgpr6, $sgpr16
	.loc	53 97 18                        ; csrc/cache_kernels_fused.hip:97:18
	v_sub_u32_e32 v3, v3, v4
	.loc	53 80 30                        ; csrc/cache_kernels_fused.hip:80:30
	v_mad_u64_u32 v[4:5], s[36:37], s34, v6, v[2:3]
	.loc	53 84 48                        ; csrc/cache_kernels_fused.hip:84:48
	v_ashrrev_i32_e32 v8, 31, v6
	.loc	53 84 57 is_stmt 0              ; csrc/cache_kernels_fused.hip:84:57
	v_mul_lo_u32 v9, s7, v6
	v_mad_u64_u32 v[6:7], s[36:37], s6, v6, 0
.Ltmp894:
	.loc	53 97 18 is_stmt 1              ; csrc/cache_kernels_fused.hip:97:18
	v_mul_lo_u32 v3, s16, v3
	.loc	53 80 30                        ; csrc/cache_kernels_fused.hip:80:30
	v_ashrrev_i32_e32 v5, 31, v4
	.loc	53 84 57                        ; csrc/cache_kernels_fused.hip:84:57
	v_mul_lo_u32 v10, s6, v8
	.loc	53 97 18                        ; csrc/cache_kernels_fused.hip:97:18
	v_add3_u32 v8, v3, s16, v2
.Ltmp895:
	;DEBUG_VALUE: x_src <- [DW_OP_LLVM_poisoned] undef
	.loc	53 80 30                        ; csrc/cache_kernels_fused.hip:80:30
	v_lshlrev_b64 v[4:5], 1, v[4:5]
	.loc	53 84 57                        ; csrc/cache_kernels_fused.hip:84:57
	v_add3_u32 v7, v7, v10, v9
	.loc	53 98 18                        ; csrc/cache_kernels_fused.hip:98:18
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	53 80 30                        ; csrc/cache_kernels_fused.hip:80:30
	v_lshl_add_u64 v[10:11], s[24:25], 0, v[4:5]
	.loc	53 84 46                        ; csrc/cache_kernels_fused.hip:84:46
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[4:5]
.Ltmp896:
	;DEBUG_VALUE: q_pe_head_ptr <- undef
	.loc	53 81 41                        ; csrc/cache_kernels_fused.hip:81:41
	v_lshl_add_u64 v[12:13], s[16:17], 1, v[10:11]
	.loc	53 97 18                        ; csrc/cache_kernels_fused.hip:97:18
	v_lshl_add_u64 v[4:5], v[6:7], 0, v[4:5]
	.loc	53 98 18                        ; csrc/cache_kernels_fused.hip:98:18
	v_lshl_add_u64 v[6:7], v[8:9], 1, v[6:7]
	.loc	53 80 16                        ; csrc/cache_kernels_fused.hip:80:16
	global_load_ushort v3, v[10:11], off
.Ltmp897:
	;DEBUG_VALUE: cos <- [DW_OP_LLVM_poisoned] $vgpr3
	.loc	53 81 16                        ; csrc/cache_kernels_fused.hip:81:16
	global_load_ushort v8, v[12:13], off
.Ltmp898:
	;DEBUG_VALUE: sin <- [DW_OP_LLVM_poisoned] $vgpr8
	.loc	53 98 18                        ; csrc/cache_kernels_fused.hip:98:18
	global_load_ushort v9, v[6:7], off
.Ltmp899:
	;DEBUG_VALUE: y_src <- [DW_OP_LLVM_poisoned] $vgpr9
	;DEBUG_VALUE: operator():this <- undef
	;DEBUG_VALUE: __hmul:x <- undef
	;DEBUG_VALUE: __hmul:y <- undef
	;DEBUG_VALUE: __hmul:x <- undef
	;DEBUG_VALUE: __hmul:y <- undef
	;DEBUG_VALUE: __hsub:y <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: __hsub:x <- undef
	;DEBUG_VALUE: x_dst <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: operator():this <- undef
	;DEBUG_VALUE: __hmul:x <- undef
	;DEBUG_VALUE: __hmul:y <- undef
	;DEBUG_VALUE: __hadd:x <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: __hadd:y <- undef
	;DEBUG_VALUE: y_dst <- [DW_OP_LLVM_poisoned] undef
	.loc	53 97 18                        ; csrc/cache_kernels_fused.hip:97:18
	s_nop 0
	global_load_ushort v10, v[4:5], off
.Ltmp900:
	;DEBUG_VALUE: x_src <- [DW_OP_LLVM_poisoned] $vgpr10
	.loc	53 74 39                        ; csrc/cache_kernels_fused.hip:74:39
	v_add_u32_e32 v2, s33, v2
.Ltmp901:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr2
	.loc	53 74 31 is_stmt 0              ; csrc/cache_kernels_fused.hip:74:31
	v_cmp_le_i32_e32 vcc, s30, v2
.Ltmp902:
	.loc	53 74 3                         ; csrc/cache_kernels_fused.hip:74:3
	s_or_b64 s[26:27], vcc, s[26:27]
.Ltmp903:
	.loc	17 1396 53 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:1396:53
	s_waitcnt vmcnt(1)
	v_mul_f16_e32 v11, v8, v9
.Ltmp904:
	;DEBUG_VALUE: __hsub:y <- [DW_OP_LLVM_poisoned] $vgpr11
	.loc	17 1396 53 is_stmt 0            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:1396:53
	v_mul_f16_e32 v9, v3, v9
.Ltmp905:
	;DEBUG_VALUE: __hadd:x <- [DW_OP_LLVM_poisoned] $vgpr9
	;DEBUG_VALUE: __hmul:x <- undef
	;DEBUG_VALUE: __hmul:y <- undef
	.loc	17 1388 53 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:1388:53
	s_waitcnt vmcnt(0)
	v_fma_f16 v3, v3, v10, -v11
.Ltmp906:
	;DEBUG_VALUE: x_dst <- [DW_OP_LLVM_poisoned] $vgpr3
	.loc	17 1368 53                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:1368:53
	v_fma_f16 v8, v8, v10, v9
.Ltmp907:
	;DEBUG_VALUE: y_dst <- [DW_OP_LLVM_poisoned] $vgpr8
	.loc	53 115 31                       ; csrc/cache_kernels_fused.hip:115:31
	global_store_short v[4:5], v3, off
	.loc	53 116 31                       ; csrc/cache_kernels_fused.hip:116:31
	global_store_short v[6:7], v8, off
.Ltmp908:
	.loc	53 74 3                         ; csrc/cache_kernels_fused.hip:74:3
	s_andn2_b64 exec, exec, s[26:27]
	s_cbranch_execnz .LBB8_2
.Ltmp909:
.LBB8_3:                                ; %Flow181
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr10_sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr30
	.loc	53 0 3 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:3
	s_or_b64 exec, exec, s[22:23]
.Ltmp910:
	.loc	54 258 116 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dword s22, s[0:1], 0x74
.Ltmp911:
	.loc	53 119 28                       ; csrc/cache_kernels_fused.hip:119:28
	s_lshl_b64 s[4:5], s[2:3], 3
.Ltmp912:
	.loc	54 258 116                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s23, s22, 31
.Ltmp913:
	.loc	53 119 28                       ; csrc/cache_kernels_fused.hip:119:28
	s_add_u32 s4, s10, s4
	s_addc_u32 s5, s11, s5
	s_load_dwordx2 s[24:25], s[4:5], 0x0
.Ltmp914:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr24_sgpr25
	.loc	53 120 38                       ; csrc/cache_kernels_fused.hip:120:38
	s_waitcnt lgkmcnt(0)
	s_or_b64 s[4:5], s[24:25], s[22:23]
	s_mov_b32 s4, 0
	s_cmp_lg_u64 s[4:5], 0
	s_cbranch_scc0 .LBB8_14
.Ltmp915:
; %bb.4:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr10_sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr30
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr24_sgpr25
	s_add_u32 s4, s22, s23
	s_mov_b32 s10, s23
.Ltmp916:
	s_mov_b32 s11, s23
	s_addc_u32 s5, s23, s23
	s_xor_b64 s[26:27], s[4:5], s[10:11]
	v_cvt_f32_u32_e32 v1, s26
	v_cvt_f32_u32_e32 v2, s27
	s_sub_u32 s3, 0, s26
	s_subb_u32 s4, 0, s27
	v_fmamk_f32 v1, v2, 0x4f800000, v1
	v_rcp_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v2, 0x2f800000, v1
	v_trunc_f32_e32 v2, v2
	v_fmamk_f32 v1, v2, 0xcf800000, v1
	v_cvt_u32_f32_e32 v2, v2
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s5, v2
	v_readfirstlane_b32 s17, v1
	s_mul_i32 s30, s3, s5
.Ltmp917:
	s_mul_hi_u32 s33, s3, s17
	s_mul_i32 s31, s4, s17
	s_add_i32 s30, s33, s30
	s_add_i32 s30, s30, s31
	s_mul_i32 s34, s3, s17
	s_mul_hi_u32 s31, s17, s30
	s_mul_i32 s33, s17, s30
	s_mul_hi_u32 s17, s17, s34
	s_add_u32 s17, s17, s33
	s_addc_u32 s31, 0, s31
	s_mul_hi_u32 s35, s5, s34
	s_mul_i32 s34, s5, s34
	s_add_u32 s17, s17, s34
	s_mul_hi_u32 s33, s5, s30
	s_addc_u32 s17, s31, s35
	s_addc_u32 s31, s33, 0
	s_mul_i32 s30, s5, s30
	s_add_u32 s17, s17, s30
	s_addc_u32 s30, 0, s31
	v_add_co_u32_e32 v1, vcc, s17, v1
	s_cmp_lg_u64 vcc, 0
	s_addc_u32 s5, s5, s30
	v_readfirstlane_b32 s30, v1
	s_mul_i32 s17, s3, s5
	s_mul_hi_u32 s31, s3, s30
	s_add_i32 s17, s31, s17
	s_mul_i32 s4, s4, s30
	s_add_i32 s17, s17, s4
	s_mul_i32 s3, s3, s30
	s_mul_hi_u32 s31, s5, s3
	s_mul_i32 s33, s5, s3
	s_mul_i32 s35, s30, s17
	s_mul_hi_u32 s3, s30, s3
	s_mul_hi_u32 s34, s30, s17
	s_add_u32 s3, s3, s35
	s_addc_u32 s30, 0, s34
	s_add_u32 s3, s3, s33
	s_mul_hi_u32 s4, s5, s17
	s_addc_u32 s3, s30, s31
	s_addc_u32 s4, s4, 0
	s_mul_i32 s17, s5, s17
	s_add_u32 s3, s3, s17
	s_addc_u32 s4, 0, s4
	v_add_co_u32_e32 v1, vcc, s3, v1
	s_cmp_lg_u64 vcc, 0
	s_addc_u32 s3, s5, s4
	s_ashr_i32 s30, s25, 31
	s_add_u32 s4, s24, s30
	s_mov_b32 s31, s30
	s_addc_u32 s5, s25, s30
	s_xor_b64 s[34:35], s[4:5], s[30:31]
	v_readfirstlane_b32 s17, v1
	s_mul_i32 s5, s34, s3
	s_mul_hi_u32 s33, s34, s17
	s_mul_hi_u32 s4, s34, s3
	s_add_u32 s5, s33, s5
	s_addc_u32 s4, 0, s4
	s_mul_hi_u32 s36, s35, s17
	s_mul_i32 s17, s35, s17
	s_add_u32 s5, s5, s17
	s_mul_hi_u32 s33, s35, s3
	s_addc_u32 s4, s4, s36
	s_addc_u32 s5, s33, 0
	s_mul_i32 s3, s35, s3
	s_add_u32 s3, s4, s3
	s_addc_u32 s17, 0, s5
	s_mul_i32 s4, s26, s17
	s_mul_hi_u32 s5, s26, s3
	s_add_i32 s4, s5, s4
	s_mul_i32 s5, s27, s3
	s_add_i32 s33, s4, s5
	s_mul_i32 s5, s26, s3
	v_mov_b32_e32 v1, s5
	s_sub_i32 s4, s35, s33
	v_sub_co_u32_e32 v1, vcc, s34, v1
	s_cmp_lg_u64 vcc, 0
	s_subb_u32 s34, s4, s27
	v_subrev_co_u32_e64 v2, s[4:5], s26, v1
	s_cmp_lg_u64 s[4:5], 0
	s_subb_u32 s4, s34, 0
	s_cmp_ge_u32 s4, s27
	v_readfirstlane_b32 s34, v2
	s_cselect_b32 s5, -1, 0
	s_cmp_ge_u32 s34, s26
	s_cselect_b32 s34, -1, 0
	s_cmp_eq_u32 s4, s27
	s_cselect_b32 s4, s34, s5
	s_add_u32 s5, s3, 1
	s_addc_u32 s34, s17, 0
	s_add_u32 s36, s3, 2
	s_addc_u32 s37, s17, 0
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s4, s36, s5
	s_cselect_b32 s5, s37, s34
	s_cmp_lg_u64 vcc, 0
	s_subb_u32 s33, s35, s33
	s_cmp_ge_u32 s33, s27
	v_readfirstlane_b32 s35, v1
	s_cselect_b32 s34, -1, 0
	s_cmp_ge_u32 s35, s26
	s_cselect_b32 s26, -1, 0
	s_cmp_eq_u32 s33, s27
	s_cselect_b32 s26, s26, s34
	s_cmp_lg_u32 s26, 0
	s_cselect_b32 s5, s5, s17
	s_cselect_b32 s4, s4, s3
	s_xor_b64 s[10:11], s[30:31], s[10:11]
	s_xor_b64 s[4:5], s[4:5], s[10:11]
	s_sub_u32 s4, s4, s10
	s_subb_u32 s5, s5, s11
	s_cbranch_execnz .LBB8_6
.Ltmp918:
.LBB8_5:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr24_sgpr25
	.loc	53 120 38                       ; csrc/cache_kernels_fused.hip:120:38
	v_cvt_f32_u32_e32 v1, s22
	s_sub_i32 s3, 0, s22
	s_mov_b32 s5, 0
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	s_nop 0
	v_readfirstlane_b32 s4, v1
	s_mul_i32 s3, s3, s4
	s_mul_hi_u32 s3, s4, s3
	s_add_i32 s4, s4, s3
	s_mul_hi_u32 s3, s24, s4
	s_mul_i32 s6, s3, s22
	s_sub_i32 s6, s24, s6
	s_add_i32 s4, s3, 1
	s_sub_i32 s7, s6, s22
	s_cmp_ge_u32 s6, s22
	s_cselect_b32 s3, s4, s3
	s_cselect_b32 s6, s7, s6
	s_add_i32 s4, s3, 1
	s_cmp_ge_u32 s6, s22
	s_cselect_b32 s4, s4, s3
.Ltmp919:
.LBB8_6:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr24_sgpr25
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_idx <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_idx <- [DW_OP_LLVM_poisoned] undef
	.loc	53 124 16                       ; csrc/cache_kernels_fused.hip:124:16
	v_cmp_lt_i64_e64 s[6:7], s[24:25], 0
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB8_13
.Ltmp920:
; %bb.7:                                ; %.preheader138
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr24_sgpr25
	.loc	53 121 38                       ; csrc/cache_kernels_fused.hip:121:38
	s_mul_i32 s3, s4, s23
	s_mul_hi_u32 s6, s4, s22
	s_add_i32 s3, s6, s3
	s_mul_i32 s6, s5, s22
	s_add_i32 s7, s3, s6
.Ltmp921:
	.loc	54 258 116                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dwordx2 s[10:11], s[0:1], 0x68
.Ltmp922:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr10
	s_load_dword s6, s[0:1], 0x70
.Ltmp923:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr6
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	53 121 38                       ; csrc/cache_kernels_fused.hip:121:38
	s_mul_i32 s3, s4, s22
	s_sub_u32 s3, s24, s3
	s_subb_u32 s24, s25, s7
.Ltmp924:
	.loc	53 129 31                       ; csrc/cache_kernels_fused.hip:129:31
	v_cmp_gt_i32_e32 vcc, s16, v0
.Ltmp925:
	.loc	53 129 3 is_stmt 0              ; csrc/cache_kernels_fused.hip:129:3
	s_and_saveexec_b64 s[22:23], vcc
	s_cbranch_execz .LBB8_10
.Ltmp926:
; %bb.8:                                ; %.lr.ph141
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr10
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr6
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 258 116 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dwordx2 s[26:27], s[0:1], 0x40
	s_load_dword s25, s[0:1], 0x8c
	s_ashr_i32 s17, s16, 31
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s30, s10, 31
	s_ashr_i32 s33, s11, 31
.Ltmp927:
	.loc	53 129 3                        ; csrc/cache_kernels_fused.hip:129:3
	s_mul_i32 s27, s27, s2
	s_mul_hi_u32 s31, s26, s2
	s_add_i32 s27, s31, s27
	s_mul_i32 s26, s26, s2
	s_ashr_i32 s7, s6, 31
	s_and_b32 s25, s25, 0xffff
	s_lshl_b64 s[26:27], s[26:27], 1
	s_add_u32 s12, s12, s26
.Ltmp928:
	s_mul_hi_u32 s31, s4, s10
	s_mul_i32 s30, s4, s30
	s_mul_hi_u32 s34, s3, s11
	s_mul_i32 s33, s3, s33
	s_addc_u32 s13, s13, s27
	s_add_i32 s30, s31, s30
	s_mul_i32 s31, s5, s10
	s_add_i32 s33, s34, s33
	s_mul_i32 s34, s24, s11
	s_add_i32 s31, s30, s31
	s_mul_i32 s30, s4, s10
	s_add_i32 s35, s33, s34
	s_mul_i32 s34, s3, s11
	s_lshl_b32 s26, s25, 1
	s_lshl_b64 s[30:31], s[30:31], 1
	s_lshl_b64 s[34:35], s[34:35], 1
	s_add_u32 s33, s30, s34
	s_addc_u32 s34, s31, s35
	s_lshl_b64 s[30:31], s[6:7], 1
	s_add_u32 s7, s33, s30
	s_addc_u32 s31, s34, s31
	s_add_u32 s30, s8, s7
	s_mul_i32 s7, s20, s29
	s_mul_hi_u32 s29, s20, s28
	s_addc_u32 s31, s9, s31
	s_add_i32 s7, s29, s7
	s_mul_i32 s21, s21, s28
.Ltmp929:
	s_add_i32 s21, s7, s21
	s_mul_i32 s20, s20, s28
	s_lshl_b64 s[20:21], s[20:21], 1
	s_add_u32 s28, s18, s20
.Ltmp930:
	v_lshlrev_b32_e32 v12, 1, v0
	v_mov_b32_e32 v13, 0
	s_addc_u32 s29, s19, s21
	v_lshl_add_u64 v[6:7], s[28:29], 0, v[12:13]
	s_lshl_b64 s[28:29], s[16:17], 1
	s_add_u32 s7, s20, s28
	v_add_u32_e32 v2, s16, v0
	s_addc_u32 s17, s21, s29
	v_ashrrev_i32_e32 v3, 31, v2
	s_add_u32 s18, s18, s7
.Ltmp931:
	v_lshlrev_b64 v[4:5], 1, v[2:3]
	s_addc_u32 s19, s19, s17
	v_lshl_add_u64 v[2:3], s[12:13], 0, v[4:5]
	s_mov_b32 s27, 0
	v_lshl_add_u64 v[4:5], s[30:31], 0, v[4:5]
	v_lshl_add_u64 v[8:9], s[18:19], 0, v[12:13]
	v_lshl_add_u64 v[10:11], s[12:13], 0, v[12:13]
	v_lshl_add_u64 v[12:13], s[30:31], 0, v[12:13]
	s_mov_b64 s[12:13], 0
	s_mov_b64 s[18:19], 0
	v_mov_b32_e32 v1, v0
.Ltmp932:
.LBB8_9:                                ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr10
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr6
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr1
	;DEBUG_VALUE: pair_idx <- [DW_OP_LLVM_poisoned] $vgpr1
	.loc	53 133 16                       ; csrc/cache_kernels_fused.hip:133:16
	v_lshl_add_u64 v[16:17], v[8:9], 0, s[18:19]
	.loc	53 132 16                       ; csrc/cache_kernels_fused.hip:132:16
	v_lshl_add_u64 v[14:15], v[6:7], 0, s[18:19]
.Ltmp933:
	;DEBUG_VALUE: cos <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: sin <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: k_pe_head_ptr <- undef
	;DEBUG_VALUE: pair_idx_x <- [DW_OP_LLVM_poisoned] $vgpr1
	;DEBUG_VALUE: pair_idx_y <- [DW_OP_LLVM_poisoned] $vgpr1, $sgpr16
	.loc	53 148 18                       ; csrc/cache_kernels_fused.hip:148:18
	v_lshl_add_u64 v[18:19], v[10:11], 0, s[18:19]
.Ltmp934:
	;DEBUG_VALUE: x_src <- [DW_OP_LLVM_poisoned] undef
	.loc	53 149 18                       ; csrc/cache_kernels_fused.hip:149:18
	v_lshl_add_u64 v[20:21], v[2:3], 0, s[18:19]
	.loc	53 133 16                       ; csrc/cache_kernels_fused.hip:133:16
	global_load_ushort v22, v[16:17], off
.Ltmp935:
	;DEBUG_VALUE: sin <- [DW_OP_LLVM_poisoned] $vgpr22
	.loc	53 149 18                       ; csrc/cache_kernels_fused.hip:149:18
	global_load_ushort v23, v[20:21], off
.Ltmp936:
	;DEBUG_VALUE: y_src <- [DW_OP_LLVM_poisoned] $vgpr23
	;DEBUG_VALUE: operator():this <- undef
	;DEBUG_VALUE: __hmul:x <- undef
	;DEBUG_VALUE: __hmul:y <- undef
	;DEBUG_VALUE: __hmul:x <- undef
	;DEBUG_VALUE: __hmul:y <- undef
	;DEBUG_VALUE: __hsub:y <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: __hsub:x <- undef
	;DEBUG_VALUE: x_dst <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: operator():this <- undef
	;DEBUG_VALUE: __hmul:x <- undef
	;DEBUG_VALUE: __hmul:y <- undef
	;DEBUG_VALUE: __hadd:x <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: __hadd:y <- undef
	;DEBUG_VALUE: y_dst <- [DW_OP_LLVM_poisoned] undef
	.loc	53 132 16                       ; csrc/cache_kernels_fused.hip:132:16
	global_load_ushort v24, v[14:15], off
.Ltmp937:
	;DEBUG_VALUE: cos <- [DW_OP_LLVM_poisoned] $vgpr24
	.loc	53 148 18                       ; csrc/cache_kernels_fused.hip:148:18
	global_load_ushort v25, v[18:19], off
.Ltmp938:
	;DEBUG_VALUE: x_src <- [DW_OP_LLVM_poisoned] $vgpr25
	.loc	53 179 36                       ; csrc/cache_kernels_fused.hip:179:36
	v_lshl_add_u64 v[14:15], v[12:13], 0, s[18:19]
	.loc	53 180 36                       ; csrc/cache_kernels_fused.hip:180:36
	v_lshl_add_u64 v[16:17], v[4:5], 0, s[18:19]
.Ltmp939:
	.loc	53 129 46                       ; csrc/cache_kernels_fused.hip:129:46
	v_add_u32_e32 v1, s25, v1
.Ltmp940:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr1
	.loc	53 129 31 is_stmt 0             ; csrc/cache_kernels_fused.hip:129:31
	s_add_u32 s18, s18, s26
	s_addc_u32 s19, s19, s27
	v_cmp_le_i32_e32 vcc, s16, v1
.Ltmp941:
	.loc	53 129 3                        ; csrc/cache_kernels_fused.hip:129:3
	s_or_b64 s[12:13], vcc, s[12:13]
.Ltmp942:
	.loc	17 1396 53 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:1396:53
	s_waitcnt vmcnt(2)
	v_mul_f16_e32 v26, v22, v23
.Ltmp943:
	;DEBUG_VALUE: __hsub:y <- [DW_OP_LLVM_poisoned] $vgpr26
	.loc	17 1396 53 is_stmt 0            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:1396:53
	s_waitcnt vmcnt(1)
	v_mul_f16_e32 v23, v24, v23
.Ltmp944:
	;DEBUG_VALUE: __hadd:x <- [DW_OP_LLVM_poisoned] $vgpr23
	;DEBUG_VALUE: __hmul:x <- undef
	;DEBUG_VALUE: __hmul:y <- undef
	.loc	17 1388 53 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:1388:53
	s_waitcnt vmcnt(0)
	v_fma_f16 v24, v24, v25, -v26
.Ltmp945:
	;DEBUG_VALUE: x_dst <- [DW_OP_LLVM_poisoned] $vgpr24
	.loc	17 1368 53                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:1368:53
	v_fma_f16 v22, v22, v25, v23
.Ltmp946:
	;DEBUG_VALUE: y_dst <- [DW_OP_LLVM_poisoned] $vgpr22
	.loc	53 166 31                       ; csrc/cache_kernels_fused.hip:166:31
	global_store_short v[18:19], v24, off
	.loc	53 167 31                       ; csrc/cache_kernels_fused.hip:167:31
	global_store_short v[20:21], v22, off
.Ltmp947:
	;DEBUG_VALUE: kv_cache_ptr <- undef
	.loc	53 179 36                       ; csrc/cache_kernels_fused.hip:179:36
	global_store_short v[14:15], v24, off
	.loc	53 180 36                       ; csrc/cache_kernels_fused.hip:180:36
	global_store_short v[16:17], v22, off
.Ltmp948:
	.loc	53 129 3                        ; csrc/cache_kernels_fused.hip:129:3
	s_andn2_b64 exec, exec, s[12:13]
	s_cbranch_execnz .LBB8_9
.Ltmp949:
.LBB8_10:                               ; %Flow177
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr10
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr6
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	.loc	53 0 3 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:3
	s_or_b64 exec, exec, s[22:23]
.Ltmp950:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	53 226 31 is_stmt 1             ; csrc/cache_kernels_fused.hip:226:31
	s_waitcnt lgkmcnt(0)
	v_cmp_gt_i32_e32 vcc, s6, v0
.Ltmp951:
	.loc	53 226 3 is_stmt 0              ; csrc/cache_kernels_fused.hip:226:3
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB8_13
.Ltmp952:
; %bb.11:                               ; %.lr.ph143
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr10
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr6
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	.loc	54 258 116 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dwordx2 s[12:13], s[0:1], 0x48
	s_load_dword s7, s[0:1], 0x8c
	s_mul_i32 s5, s5, s10
	s_mul_i32 s24, s24, s11
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s1, s13, s2
	s_mul_hi_u32 s13, s12, s2
	s_mul_i32 s0, s12, s2
	s_add_i32 s1, s13, s1
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	s_ashr_i32 s2, s10, 31
	s_mul_hi_u32 s12, s4, s10
	s_mul_i32 s2, s4, s2
	s_add_i32 s2, s12, s2
	s_add_i32 s5, s2, s5
	s_mul_i32 s4, s4, s10
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s8, s8, s4
.Ltmp953:
	.loc	54 0 116 is_stmt 0              ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:0:116
	s_addc_u32 s9, s9, s5
	s_ashr_i32 s2, s11, 31
	s_mul_hi_u32 s4, s3, s11
	s_mul_i32 s2, s3, s2
	s_add_i32 s2, s4, s2
	s_add_i32 s5, s2, s24
	s_mul_i32 s4, s3, s11
	s_lshl_b64 s[2:3], s[4:5], 1
	s_add_u32 s2, s8, s2
	s_addc_u32 s3, s9, s3
	s_and_b32 s7, s7, 0xffff
	s_mov_b64 s[4:5], 0
.Ltmp954:
.LBB8_12:                               ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr10
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr6
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	53 227 55 is_stmt 1             ; csrc/cache_kernels_fused.hip:227:55
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[2:3], 1, v[0:1]
	v_lshl_add_u64 v[4:5], s[0:1], 0, v[2:3]
	.loc	53 227 22 is_stmt 0             ; csrc/cache_kernels_fused.hip:227:22
	global_load_ushort v1, v[4:5], off
.Ltmp955:
	;DEBUG_VALUE: src_value <- [DW_OP_LLVM_poisoned] $vgpr1
	;DEBUG_VALUE: kv_cache_ptr <- undef
	.loc	53 226 49 is_stmt 1             ; csrc/cache_kernels_fused.hip:226:49
	v_add_u32_e32 v0, s7, v0
.Ltmp956:
	.loc	53 226 31 is_stmt 0             ; csrc/cache_kernels_fused.hip:226:31
	v_cmp_le_i32_e32 vcc, s6, v0
.Ltmp957:
	.loc	53 237 11 is_stmt 1             ; csrc/cache_kernels_fused.hip:237:11
	v_lshl_add_u64 v[2:3], s[2:3], 0, v[2:3]
.Ltmp958:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	53 226 3                        ; csrc/cache_kernels_fused.hip:226:3
	s_or_b64 s[4:5], vcc, s[4:5]
.Ltmp959:
	.loc	53 237 27                       ; csrc/cache_kernels_fused.hip:237:27
	s_waitcnt vmcnt(0)
	global_store_short v[2:3], v1, off
.Ltmp960:
	.loc	53 226 3                        ; csrc/cache_kernels_fused.hip:226:3
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB8_12
.Ltmp961:
.LBB8_13:                               ; %.loopexit
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	.loc	53 263 1                        ; csrc/cache_kernels_fused.hip:263:1
	s_endpgm
.Ltmp962:
.LBB8_14:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr10_sgpr11
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr8_sgpr9
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- [DW_OP_LLVM_poisoned] $sgpr20_sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr30
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, true, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr24_sgpr25
                                        ; implicit-def: $sgpr4_sgpr5
	.loc	53 0 1 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:1
	s_branch .LBB8_5
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 384
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 27
		.amdhsa_next_free_sgpr 38
		.amdhsa_accum_offset 28
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,"axG",@progbits,_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,comdat
.Lfunc_end8:
	.size	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf, .Lfunc_end8-_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.cfi_endproc
                                        ; -- End function
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.num_vgpr, 27
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.num_agpr, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.numbered_sgpr, 38
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.private_seg_size, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.uses_vcc, 1
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.uses_flat_scratch, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_dyn_sized_stack, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_recursion, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits