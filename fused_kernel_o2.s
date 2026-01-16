; Kernel info:
; codeLenInByte = 1956
; TotalNumSgprs: 54
; NumVgprs: 17
; NumAgprs: 0
; TotalNumVgprs: 17
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 6
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 54
; NumVGPRsForWavesPerEU: 17
; AccumOffset: 20
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 4
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.text._ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,"axG",@progbits,_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,comdat
	.protected	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf ; -- Begin function _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.globl	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.p2align	8
	.type	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,@function
_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf: ; @_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
.Lfunc_begin9:
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:positions <- [DW_OP_LLVM_poisoned] undef
	.loc	53 258 116 prologue_end is_stmt 1 ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dword s24, s[0:1], 0x28
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe <- [DW_OP_LLVM_poisoned] undef
	s_load_dwordx2 s[4:5], s[0:1], 0x0
.Ltmp883:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:positions <- [DW_OP_LLVM_poisoned] $sgpr4_sgpr5
	.loc	53 0 116 is_stmt 0              ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:0:116
	s_load_dwordx4 s[12:15], s[0:1], 0x10
.Ltmp884:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	.loc	54 40 29 is_stmt 1              ; csrc/cache_kernels_fused.hip:40:29
	s_mov_b32 s3, 0
.Ltmp885:
	.loc	53 258 116                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s25, s24, 31
.Ltmp886:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:num_q_heads <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe_stride_head <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_quant_scale <- undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rot_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:token_idx <- [DW_OP_LLVM_poisoned] undef
	.loc	54 41 23                        ; csrc/cache_kernels_fused.hip:41:23
	s_lshl_b64 s[30:31], s[2:3], 3
	s_add_u32 s20, s4, s30
	s_addc_u32 s21, s5, s31
	s_load_dwordx2 s[22:23], s[20:21], 0x0
.Ltmp887:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:pos <- undef
	.loc	54 0 23 is_stmt 0               ; csrc/cache_kernels_fused.hip:0:23
	s_load_dwordx2 s[26:27], s[0:1], 0x20
.Ltmp888:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	.loc	53 258 116 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dwordx8 s[4:11], s[0:1], 0x30
.Ltmp889:
	.loc	53 0 116 is_stmt 0              ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:0:116
	s_load_dwordx4 s[16:19], s[0:1], 0x58
.Ltmp890:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	.loc	53 258 116                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dword s3, s[0:1], 0x50
.Ltmp891:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:num_q_heads <- [DW_OP_LLVM_poisoned] $sgpr3
	.loc	54 43 54 is_stmt 1              ; csrc/cache_kernels_fused.hip:43:54
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s20, s22, s25
	s_mul_hi_u32 s21, s22, s24
	s_add_i32 s20, s21, s20
	s_mul_i32 s21, s23, s24
	s_add_i32 s29, s20, s21
	.loc	54 45 33                        ; csrc/cache_kernels_fused.hip:45:33
	s_lshr_b32 s20, s24, 31
	.loc	54 43 54                        ; csrc/cache_kernels_fused.hip:43:54
	s_mul_i32 s28, s22, s24
.Ltmp892:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:cos_sin_ptr <- undef
	.loc	54 45 33                        ; csrc/cache_kernels_fused.hip:45:33
	s_add_i32 s24, s24, s20
.Ltmp893:
	s_ashr_i32 s24, s24, 1
.Ltmp894:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	.loc	54 48 30                        ; csrc/cache_kernels_fused.hip:48:30
	s_mul_i32 s3, s3, s24
.Ltmp895:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr3
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 49 31                        ; csrc/cache_kernels_fused.hip:49:31
	v_cmp_gt_i32_e32 vcc, s3, v0
.Ltmp896:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe_stride_token <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe_stride <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c_stride <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_size <- [DW_OP_LLVM_poisoned] undef
	.loc	54 49 3 is_stmt 0               ; csrc/cache_kernels_fused.hip:49:3
	s_and_saveexec_b64 s[20:21], vcc
	s_cbranch_execz .LBB9_3
.Ltmp897:
; %bb.1:                                ; %.lr.ph
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr3
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 0 3                          ; csrc/cache_kernels_fused.hip:0:3
	s_load_dwordx2 s[34:35], s[0:1], 0x8
.Ltmp898:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:q_pe <- [DW_OP_LLVM_poisoned] $sgpr34_sgpr35
	s_load_dword s36, s[0:1], 0x8c
	.loc	54 43 48 is_stmt 1              ; csrc/cache_kernels_fused.hip:43:48
	s_lshl_b64 s[22:23], s[28:29], 1
	s_add_u32 s22, s26, s22
	s_mul_i32 s5, s5, s2
	s_mul_hi_u32 s33, s4, s2
	s_addc_u32 s23, s27, s23
	s_add_i32 s5, s33, s5
	s_mul_i32 s4, s4, s2
	s_ashr_i32 s25, s24, 31
	s_lshl_b64 s[4:5], s[4:5], 1
	s_waitcnt lgkmcnt(0)
	s_add_u32 s4, s34, s4
	s_addc_u32 s5, s35, s5
	s_abs_i32 s33, s24
	v_cvt_f32_u32_e32 v1, s33
	s_sub_i32 s34, 0, s33
.Ltmp899:
	.loc	54 0 48 is_stmt 0               ; csrc/cache_kernels_fused.hip:0:48
	s_and_b32 s36, s36, 0xffff
.Ltmp900:
	.loc	54 55 30 is_stmt 1              ; csrc/cache_kernels_fused.hip:55:30
	s_sub_i32 s37, 0, s24
	v_rcp_iflag_f32_e32 v1, v1
.Ltmp901:
	.loc	54 49 3                         ; csrc/cache_kernels_fused.hip:49:3
	s_lshl_b32 s39, s36, 1
	v_mov_b32_e32 v4, v0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v2, s34, v1
	v_mul_hi_u32 v2, v1, v2
.Ltmp902:
	.loc	54 72 18                        ; csrc/cache_kernels_fused.hip:72:18
	s_lshl_b32 s34, s24, 1
	v_add_u32_e32 v1, v1, v2
	s_sub_i32 s38, 0, s34
.Ltmp903:
	.loc	54 49 3                         ; csrc/cache_kernels_fused.hip:49:3
	v_lshlrev_b32_e32 v2, 1, v0
	s_mov_b64 s[34:35], 0
.Ltmp904:
.LBB9_2:                                ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr3
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr4
	.loc	54 50 22                        ; csrc/cache_kernels_fused.hip:50:22
	v_sub_u32_e32 v5, 0, v4
	v_max_i32_e32 v5, v4, v5
	v_mul_hi_u32 v6, v5, v1
	v_mul_lo_u32 v7, v6, s33
	v_sub_u32_e32 v5, v5, v7
	v_add_u32_e32 v8, 1, v6
	v_cmp_le_u32_e32 vcc, s33, v5
	v_subrev_u32_e32 v7, s33, v5
	v_ashrrev_i32_e32 v3, 31, v4
	v_cndmask_b32_e32 v6, v6, v8, vcc
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_add_u32_e32 v7, 1, v6
	v_cmp_le_u32_e32 vcc, s33, v5
	v_xor_b32_e32 v3, s25, v3
	s_nop 0
	v_cndmask_b32_e32 v5, v6, v7, vcc
	v_xor_b32_e32 v5, v5, v3
	v_sub_u32_e32 v3, v5, v3
.Ltmp905:
	;DEBUG_VALUE: head_idx <- [DW_OP_LLVM_poisoned] $vgpr3
	;DEBUG_VALUE: pair_idx <- [DW_OP_LLVM_poisoned] $vgpr4, $vgpr3, $sgpr24
	.loc	54 55 30                        ; csrc/cache_kernels_fused.hip:55:30
	v_mad_u64_u32 v[6:7], s[40:41], s37, v3, v[4:5]
.Ltmp906:
	;DEBUG_VALUE: cos <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: sin <- [DW_OP_LLVM_poisoned] undef
	.loc	54 59 48                        ; csrc/cache_kernels_fused.hip:59:48
	v_ashrrev_i32_e32 v5, 31, v3
	.loc	54 59 57 is_stmt 0              ; csrc/cache_kernels_fused.hip:59:57
	v_mul_lo_u32 v12, s7, v3
	v_mad_u64_u32 v[8:9], s[40:41], s6, v3, 0
.Ltmp907:
	;DEBUG_VALUE: pair_idx_x <- [DW_OP_LLVM_poisoned] $vgpr4, $vgpr3, $sgpr24
	;DEBUG_VALUE: pair_idx_y <- [DW_OP_LLVM_poisoned] $vgpr4, $vgpr3, $sgpr24
	.loc	54 72 18 is_stmt 1              ; csrc/cache_kernels_fused.hip:72:18
	v_mad_u64_u32 v[10:11], s[40:41], s38, v3, v[2:3]
	.loc	54 59 57                        ; csrc/cache_kernels_fused.hip:59:57
	v_mul_lo_u32 v3, s6, v5
.Ltmp908:
	.loc	54 55 30                        ; csrc/cache_kernels_fused.hip:55:30
	v_ashrrev_i32_e32 v7, 31, v6
	.loc	54 59 57                        ; csrc/cache_kernels_fused.hip:59:57
	v_add3_u32 v9, v9, v3, v12
	.loc	54 72 18                        ; csrc/cache_kernels_fused.hip:72:18
	v_ashrrev_i32_e32 v11, 31, v10
	.loc	54 55 30                        ; csrc/cache_kernels_fused.hip:55:30
	v_lshl_add_u64 v[6:7], v[6:7], 1, s[22:23]
	.loc	54 59 46                        ; csrc/cache_kernels_fused.hip:59:46
	v_lshl_add_u64 v[8:9], v[8:9], 1, s[4:5]
.Ltmp909:
	;DEBUG_VALUE: q_pe_head_ptr <- undef
	.loc	54 56 41                        ; csrc/cache_kernels_fused.hip:56:41
	v_lshl_add_u64 v[12:13], s[24:25], 1, v[6:7]
	.loc	54 55 16                        ; csrc/cache_kernels_fused.hip:55:16
	global_load_ushort v3, v[6:7], off
.Ltmp910:
	;DEBUG_VALUE: cos <- [DW_OP_LLVM_poisoned] $vgpr3
	.loc	54 56 16                        ; csrc/cache_kernels_fused.hip:56:16
	global_load_ushort v5, v[12:13], off
.Ltmp911:
	;DEBUG_VALUE: sin <- [DW_OP_LLVM_poisoned] $vgpr5
	.loc	54 72 18                        ; csrc/cache_kernels_fused.hip:72:18
	v_lshl_add_u64 v[6:7], v[10:11], 1, v[8:9]
	global_load_dword v8, v[6:7], off
.Ltmp912:
	;DEBUG_VALUE: y_src <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: operator*:x <- undef
	;DEBUG_VALUE: operator*:y <- undef
	;DEBUG_VALUE: operator*=:this <- undef
	;DEBUG_VALUE: operator*=:x <- undef
	;DEBUG_VALUE: x_src <- [DW_OP_LLVM_poisoned] $vgpr8
	;DEBUG_VALUE: operator*:x <- undef
	;DEBUG_VALUE: operator*:y <- undef
	;DEBUG_VALUE: operator*=:this <- undef
	;DEBUG_VALUE: operator*=:x <- undef
	;DEBUG_VALUE: x_dst <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: operator*:x <- undef
	;DEBUG_VALUE: operator*:y <- undef
	;DEBUG_VALUE: operator*=:this <- undef
	;DEBUG_VALUE: operator*=:x <- undef
	;DEBUG_VALUE: y_dst <- [DW_OP_LLVM_poisoned] undef
	.loc	54 49 39                        ; csrc/cache_kernels_fused.hip:49:39
	v_add_u32_e32 v4, s36, v4
.Ltmp913:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr4
	.loc	54 49 31 is_stmt 0              ; csrc/cache_kernels_fused.hip:49:31
	v_cmp_le_i32_e32 vcc, s3, v4
	v_add_u32_e32 v2, s39, v2
.Ltmp914:
	.loc	54 49 3                         ; csrc/cache_kernels_fused.hip:49:3
	s_or_b64 s[34:35], vcc, s[34:35]
.Ltmp915:
	.loc	17 205 26 is_stmt 1             ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:205:26
	s_waitcnt vmcnt(0)
	v_mul_f16_sdwa v9, v5, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
.Ltmp916:
	;DEBUG_VALUE: operator-:x <- undef
	;DEBUG_VALUE: operator-:y <- undef
	;DEBUG_VALUE: operator-=:this <- undef
	;DEBUG_VALUE: operator-=:x <- undef
	.loc	17 205 26 is_stmt 0             ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:205:26
	v_mul_f16_sdwa v10, v3, v8 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
.Ltmp917:
	;DEBUG_VALUE: operator*:x <- undef
	;DEBUG_VALUE: operator*:y <- undef
	;DEBUG_VALUE: operator*=:this <- undef
	;DEBUG_VALUE: operator*=:x <- undef
	;DEBUG_VALUE: operator+:x <- undef
	;DEBUG_VALUE: operator+:y <- undef
	;DEBUG_VALUE: operator+=:this <- undef
	;DEBUG_VALUE: operator+=:x <- undef
	.loc	17 199 26 is_stmt 1             ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:199:26
	v_fma_f16 v3, v3, v8, -v9
.Ltmp918:
	;DEBUG_VALUE: x_dst <- [DW_OP_LLVM_poisoned] $vgpr3
	.loc	17 193 26                       ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:193:26
	v_fma_f16 v5, v5, v8, v10
.Ltmp919:
	;DEBUG_VALUE: y_dst <- [DW_OP_LLVM_poisoned] $vgpr5
	.loc	54 79 31                        ; csrc/cache_kernels_fused.hip:79:31
	v_pack_b32_f16 v3, v3, v5
.Ltmp920:
	global_store_dword v[6:7], v3, off
.Ltmp921:
	.loc	54 49 3                         ; csrc/cache_kernels_fused.hip:49:3
	s_andn2_b64 exec, exec, s[34:35]
	s_cbranch_execnz .LBB9_2
.Ltmp922:
.LBB9_3:                                ; %Flow180
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache_slot_mapping <- [DW_OP_LLVM_poisoned] $sgpr18_sgpr19
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr3
	.loc	54 0 3 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:3
	s_or_b64 exec, exec, s[20:21]
	.loc	54 82 28 is_stmt 1              ; csrc/cache_kernels_fused.hip:82:28
	s_add_u32 s4, s18, s30
.Ltmp923:
	.loc	53 258 116                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_load_dwordx4 s[20:23], s[0:1], 0x68
.Ltmp924:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	.loc	54 82 28                        ; csrc/cache_kernels_fused.hip:82:28
	s_addc_u32 s5, s19, s31
	s_load_dwordx2 s[6:7], s[4:5], 0x0
.Ltmp925:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr6_sgpr7
	.loc	54 83 40                        ; csrc/cache_kernels_fused.hip:83:40
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s19, s23, 31
.Ltmp926:
	s_mov_b32 s18, s23
	.loc	54 83 38 is_stmt 0              ; csrc/cache_kernels_fused.hip:83:38
	s_or_b64 s[4:5], s[6:7], s[18:19]
	s_mov_b32 s4, 0
	s_cmp_lg_u64 s[4:5], 0
	s_cbranch_scc0 .LBB9_14
.Ltmp927:
; %bb.4:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr3
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr6_sgpr7
	s_add_u32 s4, s18, s19
	s_mov_b32 s34, s19
	s_mov_b32 s35, s19
	s_addc_u32 s5, s19, s19
	s_xor_b64 s[36:37], s[4:5], s[34:35]
	v_cvt_f32_u32_e32 v1, s36
	v_cvt_f32_u32_e32 v2, s37
	s_sub_u32 s3, 0, s36
.Ltmp928:
	s_subb_u32 s4, 0, s37
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
	v_readfirstlane_b32 s23, v1
	s_mul_i32 s25, s3, s5
	s_mul_hi_u32 s38, s3, s23
	s_mul_i32 s33, s4, s23
	s_add_i32 s25, s38, s25
	s_add_i32 s25, s25, s33
	s_mul_i32 s39, s3, s23
	s_mul_hi_u32 s33, s23, s25
	s_mul_i32 s38, s23, s25
	s_mul_hi_u32 s23, s23, s39
	s_add_u32 s23, s23, s38
	s_addc_u32 s33, 0, s33
	s_mul_hi_u32 s40, s5, s39
	s_mul_i32 s39, s5, s39
	s_add_u32 s23, s23, s39
	s_mul_hi_u32 s38, s5, s25
	s_addc_u32 s23, s33, s40
	s_addc_u32 s33, s38, 0
	s_mul_i32 s25, s5, s25
	s_add_u32 s23, s23, s25
	s_addc_u32 s25, 0, s33
	v_add_co_u32_e32 v1, vcc, s23, v1
	s_cmp_lg_u64 vcc, 0
	s_addc_u32 s5, s5, s25
	v_readfirstlane_b32 s25, v1
	s_mul_i32 s23, s3, s5
	s_mul_hi_u32 s33, s3, s25
	s_add_i32 s23, s33, s23
	s_mul_i32 s4, s4, s25
	s_add_i32 s23, s23, s4
	s_mul_i32 s3, s3, s25
	s_mul_hi_u32 s33, s5, s3
	s_mul_i32 s38, s5, s3
	s_mul_i32 s40, s25, s23
	s_mul_hi_u32 s3, s25, s3
	s_mul_hi_u32 s39, s25, s23
	s_add_u32 s3, s3, s40
	s_addc_u32 s25, 0, s39
	s_add_u32 s3, s3, s38
	s_mul_hi_u32 s4, s5, s23
	s_addc_u32 s3, s25, s33
	s_addc_u32 s4, s4, 0
	s_mul_i32 s23, s5, s23
	s_add_u32 s3, s3, s23
	s_addc_u32 s4, 0, s4
	v_add_co_u32_e32 v1, vcc, s3, v1
	s_cmp_lg_u64 vcc, 0
	s_addc_u32 s3, s5, s4
	s_ashr_i32 s38, s7, 31
	s_add_u32 s4, s6, s38
	s_mov_b32 s39, s38
	s_addc_u32 s5, s7, s38
	s_xor_b64 s[40:41], s[4:5], s[38:39]
	v_readfirstlane_b32 s23, v1
	s_mul_i32 s5, s40, s3
	s_mul_hi_u32 s25, s40, s23
	s_mul_hi_u32 s4, s40, s3
	s_add_u32 s5, s25, s5
	s_addc_u32 s4, 0, s4
	s_mul_hi_u32 s33, s41, s23
	s_mul_i32 s23, s41, s23
	s_add_u32 s5, s5, s23
	s_mul_hi_u32 s25, s41, s3
	s_addc_u32 s4, s4, s33
	s_addc_u32 s5, s25, 0
	s_mul_i32 s3, s41, s3
	s_add_u32 s3, s4, s3
	s_addc_u32 s23, 0, s5
	s_mul_i32 s4, s36, s23
	s_mul_hi_u32 s5, s36, s3
	s_add_i32 s4, s5, s4
	s_mul_i32 s5, s37, s3
	s_add_i32 s25, s4, s5
	s_mul_i32 s5, s36, s3
	v_mov_b32_e32 v1, s5
	s_sub_i32 s4, s41, s25
	v_sub_co_u32_e32 v1, vcc, s40, v1
	s_cmp_lg_u64 vcc, 0
	s_subb_u32 s33, s4, s37
	v_subrev_co_u32_e64 v2, s[4:5], s36, v1
	s_cmp_lg_u64 s[4:5], 0
	s_subb_u32 s4, s33, 0
	s_cmp_ge_u32 s4, s37
	v_readfirstlane_b32 s33, v2
	s_cselect_b32 s5, -1, 0
	s_cmp_ge_u32 s33, s36
	s_cselect_b32 s33, -1, 0
	s_cmp_eq_u32 s4, s37
	s_cselect_b32 s4, s33, s5
	s_add_u32 s5, s3, 1
	s_addc_u32 s33, s23, 0
	s_add_u32 s40, s3, 2
	s_addc_u32 s42, s23, 0
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s4, s40, s5
	s_cselect_b32 s5, s42, s33
	s_cmp_lg_u64 vcc, 0
	s_subb_u32 s25, s41, s25
	s_cmp_ge_u32 s25, s37
	v_readfirstlane_b32 s40, v1
	s_cselect_b32 s33, -1, 0
	s_cmp_ge_u32 s40, s36
	s_cselect_b32 s36, -1, 0
	s_cmp_eq_u32 s25, s37
	s_cselect_b32 s25, s36, s33
	s_cmp_lg_u32 s25, 0
	s_cselect_b32 s5, s5, s23
	s_cselect_b32 s4, s4, s3
	s_xor_b64 s[34:35], s[38:39], s[34:35]
	s_xor_b64 s[4:5], s[4:5], s[34:35]
	s_sub_u32 s4, s4, s34
	s_subb_u32 s5, s5, s35
	s_cbranch_execnz .LBB9_6
.Ltmp929:
.LBB9_5:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr6_sgpr7
	.loc	54 83 38 is_stmt 1              ; csrc/cache_kernels_fused.hip:83:38
	v_cvt_f32_u32_e32 v1, s18
	s_sub_i32 s3, 0, s18
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
	s_mul_hi_u32 s3, s6, s4
	s_mul_i32 s23, s3, s18
	s_sub_i32 s23, s6, s23
	s_add_i32 s4, s3, 1
	s_sub_i32 s25, s23, s18
	s_cmp_ge_u32 s23, s18
	s_cselect_b32 s3, s4, s3
	s_cselect_b32 s23, s25, s23
	s_add_i32 s4, s3, 1
	s_cmp_ge_u32 s23, s18
	s_cselect_b32 s4, s4, s3
.Ltmp930:
.LBB9_6:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr6_sgpr7
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_idx <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_idx <- [DW_OP_LLVM_poisoned] undef
	.loc	54 87 16                        ; csrc/cache_kernels_fused.hip:87:16
	v_cmp_lt_i64_e64 s[30:31], s[6:7], 0
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB9_13
.Ltmp931:
; %bb.7:                                ; %.preheader128
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr6_sgpr7
	.loc	54 84 38                        ; csrc/cache_kernels_fused.hip:84:38
	s_mul_i32 s3, s4, s19
	s_mul_hi_u32 s19, s4, s18
	s_add_i32 s3, s19, s3
	s_mul_i32 s19, s5, s18
	s_add_i32 s3, s3, s19
	s_mul_i32 s18, s4, s18
	s_sub_u32 s33, s6, s18
	s_subb_u32 s3, s7, s3
.Ltmp932:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 92 31                        ; csrc/cache_kernels_fused.hip:92:31
	v_cmp_gt_i32_e32 vcc, s24, v0
	s_mul_hi_u32 s35, s4, s20
	s_mul_i32 s36, s5, s20
	s_mul_i32 s18, s4, s20
	s_mul_hi_u32 s34, s33, s21
	s_mul_i32 s5, s3, s21
	s_mul_i32 s6, s33, s21
.Ltmp933:
	.loc	54 92 3 is_stmt 0               ; csrc/cache_kernels_fused.hip:92:3
	s_and_saveexec_b64 s[30:31], vcc
	s_cbranch_execz .LBB9_10
.Ltmp934:
; %bb.8:                                ; %.lr.ph131
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 0 3                          ; csrc/cache_kernels_fused.hip:0:3
	s_load_dword s3, s[0:1], 0x8c
	s_ashr_i32 s25, s24, 31
	s_ashr_i32 s7, s20, 31
	s_ashr_i32 s37, s21, 31
	s_ashr_i32 s23, s22, 31
	s_waitcnt lgkmcnt(0)
	s_and_b32 s3, s3, 0xffff
	.loc	54 92 3                         ; csrc/cache_kernels_fused.hip:92:3
	s_lshl_b64 s[38:39], s[28:29], 1
	s_add_u32 s26, s26, s38
.Ltmp935:
	s_mul_i32 s7, s4, s7
	s_addc_u32 s27, s27, s39
	s_add_i32 s7, s35, s7
	s_add_i32 s19, s7, s36
	s_mul_i32 s7, s33, s37
	s_add_i32 s7, s34, s7
	v_lshlrev_b32_e32 v6, 1, v0
	v_mov_b32_e32 v7, 0
	s_add_i32 s7, s7, s5
	v_lshl_add_u64 v[2:3], s[26:27], 0, v[6:7]
	s_lshl_b32 s28, s3, 1
	s_lshl_b64 s[26:27], s[24:25], 1
	s_lshl_b64 s[38:39], s[18:19], 1
	s_lshl_b64 s[40:41], s[6:7], 1
	s_add_u32 s7, s38, s40
	s_addc_u32 s19, s39, s41
	s_lshl_b64 s[38:39], s[22:23], 1
	s_add_u32 s23, s16, s38
	s_addc_u32 s25, s17, s39
	s_add_u32 s38, s23, s7
	s_addc_u32 s39, s25, s19
	s_mul_i32 s9, s9, s2
	s_mul_hi_u32 s19, s8, s2
	s_add_i32 s9, s19, s9
	s_mul_i32 s8, s8, s2
	s_lshl_b32 s7, s3, 2
	s_lshl_b64 s[8:9], s[8:9], 1
	v_lshlrev_b32_e32 v6, 2, v0
	s_add_u32 s8, s12, s8
	v_lshl_add_u64 v[4:5], s[38:39], 0, v[6:7]
	s_addc_u32 s9, s13, s9
	s_mov_b32 s29, 0
	v_lshl_add_u64 v[4:5], v[4:5], 0, 2
	v_lshl_add_u64 v[6:7], s[8:9], 0, v[6:7]
	s_mov_b64 s[8:9], 0
	s_mov_b64 s[12:13], 0
.Ltmp936:
	.loc	54 0 3                          ; csrc/cache_kernels_fused.hip:0:3
	v_mov_b32_e32 v1, v0
.Ltmp937:
.LBB9_9:                                ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr1
	;DEBUG_VALUE: pair_idx <- [DW_OP_LLVM_poisoned] $vgpr1
	.loc	54 96 16 is_stmt 1              ; csrc/cache_kernels_fused.hip:96:16
	v_lshl_add_u64 v[8:9], v[2:3], 0, s[26:27]
	.loc	54 95 16                        ; csrc/cache_kernels_fused.hip:95:16
	global_load_ushort v12, v[2:3], off
.Ltmp938:
	;DEBUG_VALUE: cos <- [DW_OP_LLVM_poisoned] $vgpr12
	;DEBUG_VALUE: sin <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: k_pe_head_ptr <- undef
	;DEBUG_VALUE: pair_idx_x <- [DW_OP_LLVM_poisoned] $vgpr1
	;DEBUG_VALUE: pair_idx_y <- [DW_OP_LLVM_poisoned] $vgpr1
	.loc	54 111 18                       ; csrc/cache_kernels_fused.hip:111:18
	v_lshl_add_u64 v[10:11], v[6:7], 0, s[12:13]
	.loc	54 96 16                        ; csrc/cache_kernels_fused.hip:96:16
	global_load_ushort v13, v[8:9], off
.Ltmp939:
	;DEBUG_VALUE: sin <- [DW_OP_LLVM_poisoned] $vgpr13
	.loc	54 111 18                       ; csrc/cache_kernels_fused.hip:111:18
	global_load_dword v14, v[10:11], off
.Ltmp940:
	;DEBUG_VALUE: y_src <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: operator*:x <- undef
	;DEBUG_VALUE: operator*:y <- undef
	;DEBUG_VALUE: operator*=:this <- undef
	;DEBUG_VALUE: operator*=:x <- undef
	;DEBUG_VALUE: x_src <- [DW_OP_LLVM_poisoned] $vgpr14
	;DEBUG_VALUE: operator*:x <- undef
	;DEBUG_VALUE: operator*:y <- undef
	;DEBUG_VALUE: operator*=:this <- undef
	;DEBUG_VALUE: operator*=:x <- undef
	;DEBUG_VALUE: x_dst <- [DW_OP_LLVM_poisoned] undef
	;DEBUG_VALUE: operator*:x <- undef
	;DEBUG_VALUE: operator*:y <- undef
	;DEBUG_VALUE: operator*=:this <- undef
	;DEBUG_VALUE: operator*=:x <- undef
	;DEBUG_VALUE: y_dst <- [DW_OP_LLVM_poisoned] undef
	.loc	54 130 36                       ; csrc/cache_kernels_fused.hip:130:36
	v_lshl_add_u64 v[8:9], v[4:5], 0, s[12:13]
.Ltmp941:
	.loc	54 92 46                        ; csrc/cache_kernels_fused.hip:92:46
	v_add_u32_e32 v1, s3, v1
.Ltmp942:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr1
	.loc	54 92 31 is_stmt 0              ; csrc/cache_kernels_fused.hip:92:31
	s_add_u32 s12, s12, s7
	s_addc_u32 s13, s13, 0
	v_cmp_le_i32_e32 vcc, s24, v1
	v_lshl_add_u64 v[2:3], v[2:3], 0, s[28:29]
.Ltmp943:
	.loc	54 92 3                         ; csrc/cache_kernels_fused.hip:92:3
	s_or_b64 s[8:9], vcc, s[8:9]
.Ltmp944:
	.loc	17 205 26 is_stmt 1             ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:205:26
	s_waitcnt vmcnt(0)
	v_mul_f16_sdwa v15, v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
.Ltmp945:
	;DEBUG_VALUE: operator-:x <- undef
	;DEBUG_VALUE: operator-:y <- undef
	;DEBUG_VALUE: operator-=:this <- undef
	;DEBUG_VALUE: operator-=:x <- undef
	.loc	17 205 26 is_stmt 0             ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:205:26
	v_mul_f16_sdwa v16, v12, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
.Ltmp946:
	;DEBUG_VALUE: operator*:x <- undef
	;DEBUG_VALUE: operator*:y <- undef
	;DEBUG_VALUE: operator*=:this <- undef
	;DEBUG_VALUE: operator*=:x <- undef
	;DEBUG_VALUE: operator+:x <- undef
	;DEBUG_VALUE: operator+:y <- undef
	;DEBUG_VALUE: operator+=:this <- undef
	;DEBUG_VALUE: operator+=:x <- undef
	.loc	17 199 26 is_stmt 1             ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:199:26
	v_fma_f16 v12, v12, v14, -v15
.Ltmp947:
	;DEBUG_VALUE: x_dst <- [DW_OP_LLVM_poisoned] $vgpr12
	.loc	17 193 26                       ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_fp16.h:193:26
	v_fma_f16 v13, v13, v14, v16
.Ltmp948:
	;DEBUG_VALUE: y_dst <- [DW_OP_LLVM_poisoned] $vgpr13
	.loc	54 118 31                       ; csrc/cache_kernels_fused.hip:118:31
	v_pack_b32_f16 v12, v12, v13
.Ltmp949:
	global_store_dword v[10:11], v12, off
.Ltmp950:
	;DEBUG_VALUE: kv_cache_ptr <- undef
	.loc	54 131 36                       ; csrc/cache_kernels_fused.hip:131:36
	global_store_dword v[8:9], v12, off offset:-2
.Ltmp951:
	.loc	54 92 3                         ; csrc/cache_kernels_fused.hip:92:3
	s_andn2_b64 exec, exec, s[8:9]
	s_cbranch_execnz .LBB9_9
.Ltmp952:
.LBB9_10:                               ; %Flow176
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	.loc	54 0 3 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:3
	s_or_b64 exec, exec, s[30:31]
.Ltmp953:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 149 31 is_stmt 1             ; csrc/cache_kernels_fused.hip:149:31
	v_cmp_gt_i32_e32 vcc, s22, v0
.Ltmp954:
	.loc	54 149 3 is_stmt 0              ; csrc/cache_kernels_fused.hip:149:3
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execz .LBB9_13
.Ltmp955:
; %bb.11:                               ; %.lr.ph133
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 0 3                          ; csrc/cache_kernels_fused.hip:0:3
	s_mul_i32 s3, s11, s2
	s_mul_hi_u32 s7, s10, s2
	s_add_i32 s3, s7, s3
	s_mul_i32 s2, s10, s2
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s2, s14, s2
	s_addc_u32 s3, s15, s3
	s_ashr_i32 s7, s20, 31
	s_mul_i32 s4, s4, s7
	s_add_i32 s4, s35, s4
	s_add_i32 s19, s4, s36
	s_lshl_b64 s[8:9], s[18:19], 1
	s_add_u32 s4, s16, s8
	s_addc_u32 s8, s17, s9
	s_ashr_i32 s7, s21, 31
	s_mul_i32 s33, s33, s7
	s_load_dword s9, s[0:1], 0x8c
	s_add_i32 s7, s34, s33
	s_add_i32 s7, s7, s5
	s_lshl_b64 s[0:1], s[6:7], 1
	s_add_u32 s0, s4, s0
	s_addc_u32 s1, s8, s1
	s_waitcnt lgkmcnt(0)
	s_and_b32 s6, s9, 0xffff
	s_mov_b64 s[4:5], 0
.Ltmp956:
.LBB9_12:                               ; =>This Inner Loop Header: Depth=1
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 150 58 is_stmt 1             ; csrc/cache_kernels_fused.hip:150:58
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[2:3], 1, v[0:1]
	v_lshl_add_u64 v[4:5], s[2:3], 0, v[2:3]
.Ltmp957:
	;DEBUG_VALUE: src_ptr <- undef
	.loc	54 152 9                        ; csrc/cache_kernels_fused.hip:152:9
	global_load_ushort v1, v[4:5], off
.Ltmp958:
	;DEBUG_VALUE: src_value <- [DW_OP_LLVM_poisoned] $vgpr1
	;DEBUG_VALUE: kv_cache_ptr <- undef
	.loc	54 149 49                       ; csrc/cache_kernels_fused.hip:149:49
	v_add_u32_e32 v0, s6, v0
.Ltmp959:
	.loc	54 149 31 is_stmt 0             ; csrc/cache_kernels_fused.hip:149:31
	v_cmp_le_i32_e32 vcc, s22, v0
.Ltmp960:
	.loc	54 158 7 is_stmt 1              ; csrc/cache_kernels_fused.hip:158:7
	v_lshl_add_u64 v[2:3], s[0:1], 0, v[2:3]
.Ltmp961:
	;DEBUG_VALUE: i <- [DW_OP_LLVM_poisoned] $vgpr0
	.loc	54 149 3                        ; csrc/cache_kernels_fused.hip:149:3
	s_or_b64 s[4:5], vcc, s[4:5]
.Ltmp962:
	.loc	54 158 23                       ; csrc/cache_kernels_fused.hip:158:23
	s_waitcnt vmcnt(0)
	global_store_short v[2:3], v1, off
.Ltmp963:
	.loc	54 149 3                        ; csrc/cache_kernels_fused.hip:149:3
	s_andn2_b64 exec, exec, s[4:5]
	s_cbranch_execnz .LBB9_12
.Ltmp964:
.LBB9_13:                               ; %.loopexit
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	.loc	54 164 1                        ; csrc/cache_kernels_fused.hip:164:1
	s_endpgm
.Ltmp965:
.LBB9_14:
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_cache <- [DW_OP_LLVM_poisoned] $sgpr16_sgpr17
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:rope_cos_sin_cache <- [DW_OP_LLVM_poisoned] $sgpr26_sgpr27
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_c <- [DW_OP_LLVM_poisoned] $sgpr14_sgpr15
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:k_pe <- [DW_OP_LLVM_poisoned] $sgpr12_sgpr13
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:block_stride <- [DW_OP_LLVM_poisoned] $sgpr20
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:entry_stride <- [DW_OP_LLVM_poisoned] $sgpr21
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:kv_lora_rank <- [DW_OP_LLVM_poisoned] $sgpr22
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:embed_dim <- [DW_OP_LLVM_poisoned] $sgpr24
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:nq <- [DW_OP_LLVM_poisoned] $sgpr3
	;DEBUG_VALUE: concat_and_cache_mla_rope_fused_kernel<__half, false, unsigned short, unsigned short, (vllm::Fp8KVCacheDataType)0>:slot_idx <- [DW_OP_LLVM_poisoned] $sgpr6_sgpr7
                                        ; implicit-def: $sgpr4_sgpr5
	.loc	54 0 1 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:1
	s_branch .LBB9_5
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
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
		.amdhsa_next_free_vgpr 17
		.amdhsa_next_free_sgpr 43
		.amdhsa_accum_offset 20
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
	.section	.text._ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,"axG",@progbits,_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,comdat
.Lfunc_end9:
	.size	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf, .Lfunc_end9-_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.cfi_endproc
                                        ; -- End function
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.num_vgpr, 17
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.num_agpr, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.numbered_sgpr, 43
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.private_seg_size, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.uses_vcc, 1
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.uses_flat_scratch, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_dyn_sized_stack, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_recursion, 0
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb0EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits