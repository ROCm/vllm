; Function info:
; codeLenInByte = 1316
; TotalNumSgprs: 40
; NumVgprs: 42
; NumAgprs: 1
; TotalNumVgprs: 45
; ScratchSize: 120
; MemoryBound: 0
	.section	.text._ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,"axG",@progbits,_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,comdat
	.protected	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf ; -- Begin function _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.globl	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.p2align	8
	.type	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf,@function
_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf: ; @_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
.Lfunc_begin86:
	.loc	55 63 0 is_stmt 1               ; csrc/cache_kernels_fused.hip:63:0
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	s_mov_b32 s33, 0
	s_mov_b32 s32, 0x240
	s_mov_b32 s14, s10
                                        ; implicit-def: $vgpr42 : SGPR spill to VGPR lane
	v_writelane_b32 v42, s14, 0
	s_mov_b32 s13, s9
	v_writelane_b32 v42, s13, 1
	s_mov_b32 s12, s8
	v_writelane_b32 v42, s12, 2
	v_writelane_b32 v42, s6, 3
	s_nop 1
	v_writelane_b32 v42, s7, 4
	v_writelane_b32 v42, s4, 5
	s_nop 1
	v_writelane_b32 v42, s5, 6
	v_writelane_b32 v42, s2, 7
	s_nop 1
	v_writelane_b32 v42, s3, 8
	v_writelane_b32 v42, s0, 9
	s_nop 1
	v_writelane_b32 v42, s1, 10
	v_mov_b32_e32 v31, v0
	v_accvgpr_write_b32 a32, v31            ;  Reload Reuse
	s_load_dwordx2 s[64:65], s[4:5], 0x60
	s_load_dwordx2 s[68:69], s[4:5], 0x58
	s_load_dwordx2 s[88:89], s[4:5], 0x0
	s_load_dwordx2 s[84:85], s[4:5], 0x8
	s_load_dwordx2 s[80:81], s[4:5], 0x10
	s_load_dwordx2 s[76:77], s[4:5], 0x18
	s_load_dwordx2 s[72:73], s[4:5], 0x20
                                        ; kill: def $sgpr0_sgpr1 killed $sgpr64_sgpr65
                                        ; kill: def $sgpr0_sgpr1 killed $sgpr68_sgpr69
                                        ; kill: def $sgpr0_sgpr1 killed $sgpr72_sgpr73
                                        ; kill: def $sgpr0_sgpr1 killed $sgpr76_sgpr77
                                        ; kill: def $sgpr0_sgpr1 killed $sgpr80_sgpr81
                                        ; kill: def $sgpr0_sgpr1 killed $sgpr84_sgpr85
                                        ; kill: def $sgpr0_sgpr1 killed $sgpr88_sgpr89
	s_load_dword s46, s[4:5], 0x28
	s_load_dwordx2 s[42:43], s[4:5], 0x30
	s_load_dwordx2 s[38:39], s[4:5], 0x38
	s_load_dwordx2 s[34:35], s[4:5], 0x40
	s_load_dwordx2 s[28:29], s[4:5], 0x48
	s_load_dword s15, s[4:5], 0x50
	s_load_dword s9, s[4:5], 0x68
	s_load_dword s8, s[4:5], 0x6c
	s_load_dword s1, s[4:5], 0x70
	s_load_dword s0, s[4:5], 0x74
	s_load_dwordx2 s[60:61], s[4:5], 0x78
	s_mov_b64 s[4:5], 0
	v_writelane_b32 v42, s4, 11
	s_nop 1
	v_writelane_b32 v42, s5, 12
	s_mov_b32 s92, s5
	v_writelane_b32 v42, s92, 13
	s_mov_b64 s[2:3], src_private_base
	s_mov_b32 s6, 32
	v_writelane_b32 v42, s6, 14
	s_lshr_b64 s[6:7], s[2:3], s6
	s_mov_b32 s93, -1
	v_writelane_b32 v42, s93, 15
	s_add_i32 s2, s33, 56
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_mov_b32 s47, s6
	v_writelane_b32 v42, s47, 16
	s_cselect_b32 s2, s47, s92
	s_mov_b32 s91, s4
	v_writelane_b32 v42, s91, 17
	s_cselect_b32 s86, s3, s91
                                        ; kill: def $sgpr86 killed $sgpr86 def $sgpr86_sgpr87
	s_mov_b32 s87, s2
	s_add_i32 s2, s33, 64
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s82, s3, s91
                                        ; kill: def $sgpr82 killed $sgpr82 def $sgpr82_sgpr83
	s_mov_b32 s83, s2
	s_add_i32 s2, s33, 0x48
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s78, s3, s91
                                        ; kill: def $sgpr78 killed $sgpr78 def $sgpr78_sgpr79
	s_mov_b32 s79, s2
	s_add_i32 s2, s33, 0x50
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s74, s3, s91
                                        ; kill: def $sgpr74 killed $sgpr74 def $sgpr74_sgpr75
	s_mov_b32 s75, s2
	s_add_i32 s2, s33, 0x58
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s70, s3, s91
                                        ; kill: def $sgpr70 killed $sgpr70 def $sgpr70_sgpr71
	s_mov_b32 s71, s2
	s_add_i32 s2, s33, 0x60
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s66, s3, s91
                                        ; kill: def $sgpr66 killed $sgpr66 def $sgpr66_sgpr67
	s_mov_b32 s67, s2
	s_add_i32 s2, s33, 0x68
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s62, s3, s91
                                        ; kill: def $sgpr62 killed $sgpr62 def $sgpr62_sgpr63
	s_mov_b32 s63, s2
	s_add_i32 s2, s33, 0x70
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s58, s3, s91
                                        ; kill: def $sgpr58 killed $sgpr58 def $sgpr58_sgpr59
	s_mov_b32 s59, s2
	s_add_i32 s2, s33, 0x78
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s18, s3, s91
                                        ; kill: def $sgpr18 killed $sgpr18 def $sgpr18_sgpr19
	s_mov_b32 s19, s2
	s_add_i32 s2, s33, 0x80
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s56, s3, s91
                                        ; kill: def $sgpr56 killed $sgpr56 def $sgpr56_sgpr57
	s_mov_b32 s57, s2
	s_mov_b64 s[2:3], s[56:57]
	v_writelane_b32 v42, s2, 18
	s_nop 1
	v_writelane_b32 v42, s3, 19
	s_add_i32 s2, s33, 0x88
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s54, s3, s91
                                        ; kill: def $sgpr54 killed $sgpr54 def $sgpr54_sgpr55
	s_mov_b32 s55, s2
	s_mov_b64 s[2:3], s[54:55]
	v_writelane_b32 v42, s2, 20
	s_nop 1
	v_writelane_b32 v42, s3, 21
	s_add_i32 s2, s33, 0x90
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s52, s3, s91
                                        ; kill: def $sgpr52 killed $sgpr52 def $sgpr52_sgpr53
	s_mov_b32 s53, s2
	s_mov_b64 s[2:3], s[52:53]
	v_writelane_b32 v42, s2, 22
	s_nop 1
	v_writelane_b32 v42, s3, 23
	s_add_i32 s2, s33, 0x98
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s50, s3, s91
                                        ; kill: def $sgpr50 killed $sgpr50 def $sgpr50_sgpr51
	s_mov_b32 s51, s2
	v_writelane_b32 v42, s50, 24
	s_nop 1
	v_writelane_b32 v42, s51, 25
	s_add_i32 s2, s33, 0xa0
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s48, s3, s91
                                        ; kill: def $sgpr48 killed $sgpr48 def $sgpr48_sgpr49
	s_mov_b32 s49, s2
	v_writelane_b32 v42, s48, 26
	s_nop 1
	v_writelane_b32 v42, s49, 27
	s_add_i32 s2, s33, 0xa8
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s44, s3, s91
                                        ; kill: def $sgpr44 killed $sgpr44 def $sgpr44_sgpr45
	s_mov_b32 s45, s2
	s_mov_b64 s[2:3], s[44:45]
	v_writelane_b32 v42, s2, 28
	s_nop 1
	v_writelane_b32 v42, s3, 29
	s_add_i32 s2, s33, 0xb0
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s40, s3, s91
                                        ; kill: def $sgpr40 killed $sgpr40 def $sgpr40_sgpr41
	s_mov_b32 s41, s2
	s_mov_b64 s[2:3], s[40:41]
	v_writelane_b32 v42, s2, 30
	s_nop 1
	v_writelane_b32 v42, s3, 31
	s_add_i32 s2, s33, 0xb8
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s36, s3, s91
                                        ; kill: def $sgpr36 killed $sgpr36 def $sgpr36_sgpr37
	s_mov_b32 s37, s2
	s_mov_b64 s[2:3], s[36:37]
	v_writelane_b32 v42, s2, 32
	s_nop 1
	v_writelane_b32 v42, s3, 33
	s_add_i32 s2, s33, 0xc0
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s30, s3, s91
                                        ; kill: def $sgpr30 killed $sgpr30 def $sgpr30_sgpr31
	s_mov_b32 s31, s2
	s_mov_b64 s[2:3], s[30:31]
	v_writelane_b32 v42, s2, 34
	s_nop 1
	v_writelane_b32 v42, s3, 35
	s_add_i32 s2, s33, 0xc8
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s4, s3, s91
                                        ; kill: def $sgpr4 killed $sgpr4 def $sgpr4_sgpr5
	s_mov_b32 s5, s2
	s_add_i32 s2, s33, 0xd0
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s26, s3, s91
                                        ; kill: def $sgpr26 killed $sgpr26 def $sgpr26_sgpr27
	s_mov_b32 s27, s2
	s_mov_b64 s[2:3], s[26:27]
	v_writelane_b32 v42, s2, 36
	s_nop 1
	v_writelane_b32 v42, s3, 37
	s_add_i32 s2, s33, 0xd8
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s24, s3, s91
                                        ; kill: def $sgpr24 killed $sgpr24 def $sgpr24_sgpr25
	s_mov_b32 s25, s2
	s_mov_b64 s[2:3], s[24:25]
	v_writelane_b32 v42, s2, 38
	s_nop 1
	v_writelane_b32 v42, s3, 39
	s_add_i32 s2, s33, 0xe0
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s22, s3, s91
                                        ; kill: def $sgpr22 killed $sgpr22 def $sgpr22_sgpr23
	s_mov_b32 s23, s2
	s_mov_b64 s[2:3], s[22:23]
	v_writelane_b32 v42, s2, 40
	s_nop 1
	v_writelane_b32 v42, s3, 41
	s_add_i32 s2, s33, 0xe4
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s20, s3, s91
                                        ; kill: def $sgpr20 killed $sgpr20 def $sgpr20_sgpr21
	s_mov_b32 s21, s2
	s_mov_b64 s[2:3], s[20:21]
	v_writelane_b32 v42, s2, 42
	s_nop 1
	v_writelane_b32 v42, s3, 43
	s_add_i32 s2, s33, 0xe8
	s_mov_b32 s3, s2
	s_cmp_lg_u32 s3, s93
	s_cselect_b32 s2, s47, s92
	s_cselect_b32 s6, s3, s91
                                        ; kill: def $sgpr6 killed $sgpr6 def $sgpr6_sgpr7
	s_mov_b32 s7, s2
	s_mov_b64 s[2:3], s[6:7]
	v_writelane_b32 v42, s2, 44
	s_nop 1
	v_writelane_b32 v42, s3, 45
	s_add_i32 s3, s33, 0xec
	s_mov_b32 s2, s3
	s_cmp_lg_u32 s2, s93
	s_cselect_b32 s10, s47, s92
	s_cselect_b32 s2, s2, s91
                                        ; kill: def $sgpr2 killed $sgpr2 def $sgpr2_sgpr3
	s_mov_b32 s3, s10
	s_mov_b64 s[10:11], s[2:3]
	v_writelane_b32 v42, s10, 46
	s_nop 1
	v_writelane_b32 v42, s11, 47
	s_add_i32 s10, s33, 0xf0
	s_mov_b32 s11, s10
	s_cmp_lg_u32 s11, s93
	s_cselect_b32 s10, s47, s92
	s_cselect_b32 s11, s11, s91
	v_mov_b32_e32 v0, s11
	v_mov_b32_e32 v2, s10
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	s_add_i32 s10, s33, 0xf8
	s_mov_b32 s11, s10
	s_cmp_lg_u32 s11, s93
	s_cselect_b32 s10, s47, s92
	s_cselect_b32 s16, s11, s91
                                        ; kill: def $sgpr16 killed $sgpr16 def $sgpr16_sgpr17
	s_mov_b32 s17, s10
	s_mov_b64 s[10:11], s[16:17]
	v_writelane_b32 v42, s10, 48
	s_nop 1
	v_writelane_b32 v42, s11, 49
	s_add_i32 s11, s33, 0x100
	s_mov_b32 s10, s11
	s_cmp_lg_u32 s10, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s10, s10, s91
                                        ; kill: def $sgpr10 killed $sgpr10 def $sgpr10_sgpr11
	s_mov_b32 s11, s90
	v_writelane_b32 v42, s10, 50
	s_nop 1
	v_writelane_b32 v42, s11, 51
	s_add_i32 s11, s33, 0x108
	s_mov_b32 s10, s11
	s_cmp_lg_u32 s10, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s10, s10, s91
                                        ; kill: def $sgpr10 killed $sgpr10 def $sgpr10_sgpr11
	s_mov_b32 s11, s90
	s_mov_b64 s[94:95], s[10:11]
	v_writelane_b32 v42, s94, 52
	s_nop 1
	v_writelane_b32 v42, s95, 53
	s_add_i32 s90, s33, 0x110
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v42, s94, 54
	s_nop 1
	v_writelane_b32 v42, s95, 55
	v_writelane_b32 v42, s94, 56
	s_nop 1
	v_writelane_b32 v42, s95, 57
	s_add_i32 s90, s33, 0x114
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v42, s94, 58
	s_nop 1
	v_writelane_b32 v42, s95, 59
	v_writelane_b32 v42, s94, 60
	s_nop 1
	v_writelane_b32 v42, s95, 61
	s_add_i32 s90, s33, 0x118
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v42, s94, 62
	s_nop 1
	v_writelane_b32 v42, s95, 63
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a33, v42            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
                                        ; implicit-def: $vgpr43 : SGPR spill to VGPR lane
	v_writelane_b32 v43, s94, 0
	s_nop 1
	v_writelane_b32 v43, s95, 1
	s_add_i32 s90, s33, 0x11c
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 2
	s_nop 1
	v_writelane_b32 v43, s95, 3
	s_add_i32 s90, s33, 0x120
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 4
	s_nop 1
	v_writelane_b32 v43, s95, 5
	s_add_i32 s90, s33, 0x124
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 6
	s_nop 1
	v_writelane_b32 v43, s95, 7
	s_add_i32 s90, s33, 0x126
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 8
	s_nop 1
	v_writelane_b32 v43, s95, 9
	s_add_i32 s90, s33, 0x128
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 10
	s_nop 1
	v_writelane_b32 v43, s95, 11
	s_add_i32 s90, s33, 0x130
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 12
	s_nop 1
	v_writelane_b32 v43, s95, 13
	s_add_i32 s90, s33, 0x134
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 14
	s_nop 1
	v_writelane_b32 v43, s95, 15
	s_add_i32 s90, s33, 0x138
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 16
	s_nop 1
	v_writelane_b32 v43, s95, 17
	s_add_i32 s90, s33, 0x13a
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 18
	s_nop 1
	v_writelane_b32 v43, s95, 19
	s_add_i32 s90, s33, 0x13c
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 20
	s_nop 1
	v_writelane_b32 v43, s95, 21
	s_add_i32 s90, s33, 0x140
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 22
	s_nop 1
	v_writelane_b32 v43, s95, 23
	s_add_i32 s90, s33, 0x160
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 24
	s_nop 1
	v_writelane_b32 v43, s95, 25
	s_add_i32 s90, s33, 0x168
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 26
	s_nop 1
	v_writelane_b32 v43, s95, 27
	s_add_i32 s90, s33, 0x188
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 28
	s_nop 1
	v_writelane_b32 v43, s95, 29
	s_add_i32 s90, s33, 0x190
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 30
	s_nop 1
	v_writelane_b32 v43, s95, 31
	s_add_i32 s90, s33, 0x198
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 32
	s_nop 1
	v_writelane_b32 v43, s95, 33
	s_add_i32 s90, s33, 0x1a0
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 34
	s_nop 1
	v_writelane_b32 v43, s95, 35
	s_add_i32 s90, s33, 0x1a4
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 36
	s_nop 1
	v_writelane_b32 v43, s95, 37
	s_add_i32 s90, s33, 0x1a8
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 38
	s_nop 1
	v_writelane_b32 v43, s95, 39
	s_add_i32 s90, s33, 0x1aa
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 40
	s_nop 1
	v_writelane_b32 v43, s95, 41
	s_add_i32 s90, s33, 0x1b0
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 42
	s_nop 1
	v_writelane_b32 v43, s95, 43
	s_add_i32 s90, s33, 0x1b8
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 44
	s_nop 1
	v_writelane_b32 v43, s95, 45
	s_add_i32 s90, s33, 0x1bc
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 46
	s_nop 1
	v_writelane_b32 v43, s95, 47
	s_add_i32 s90, s33, 0x1c0
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 48
	s_nop 1
	v_writelane_b32 v43, s95, 49
	s_add_i32 s90, s33, 0x1c2
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 50
	s_nop 1
	v_writelane_b32 v43, s95, 51
	s_add_i32 s90, s33, 0x1c4
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 52
	s_nop 1
	v_writelane_b32 v43, s95, 53
	s_add_i32 s90, s33, 0x1c8
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 54
	s_nop 1
	v_writelane_b32 v43, s95, 55
	s_add_i32 s90, s33, 0x1e8
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 56
	s_nop 1
	v_writelane_b32 v43, s95, 57
	s_add_i32 s90, s33, 0x1f0
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 58
	s_nop 1
	v_writelane_b32 v43, s95, 59
	s_add_i32 s90, s33, 0x210
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 60
	s_nop 1
	v_writelane_b32 v43, s95, 61
	s_add_i32 s90, s33, 0x218
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 62
	s_nop 1
	v_writelane_b32 v43, s95, 63
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a34, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_add_i32 s90, s33, 0x21a
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
                                        ; implicit-def: $vgpr43 : SGPR spill to VGPR lane
	v_writelane_b32 v43, s94, 0
	s_nop 1
	v_writelane_b32 v43, s95, 1
	s_add_i32 s90, s33, 0x21c
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 2
	s_nop 1
	v_writelane_b32 v43, s95, 3
	s_add_i32 s90, s33, 0x220
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 4
	s_nop 1
	v_writelane_b32 v43, s95, 5
	s_add_i32 s90, s33, 0x228
	s_mov_b32 s94, s90
	s_cmp_lg_u32 s94, s93
	s_cselect_b32 s90, s47, s92
	s_cselect_b32 s94, s94, s91
                                        ; kill: def $sgpr94 killed $sgpr94 def $sgpr94_sgpr95
	s_mov_b32 s95, s90
	v_writelane_b32 v43, s94, 6
	s_nop 1
	v_writelane_b32 v43, s95, 7
	s_add_i32 s94, s33, 0x230
	s_mov_b32 s90, s94
	s_cmp_lg_u32 s90, s93
	s_cselect_b32 s47, s47, s92
	s_cselect_b32 s90, s90, s91
                                        ; kill: def $sgpr90 killed $sgpr90 def $sgpr90_sgpr91
	s_mov_b32 s91, s47
	v_writelane_b32 v43, s90, 8
	s_nop 1
	v_writelane_b32 v43, s91, 9
	v_mov_b64_e32 v[2:3], s[86:87]
	s_waitcnt lgkmcnt(0)
	v_mov_b64_e32 v[4:5], s[88:89]
	flat_store_dwordx2 v[2:3], v[4:5]
	v_mov_b64_e32 v[2:3], s[86:87]
	flat_load_dwordx2 v[18:19], v[2:3]
	v_mov_b64_e32 v[2:3], s[82:83]
	v_mov_b64_e32 v[4:5], s[84:85]
	flat_store_dwordx2 v[2:3], v[4:5]
	v_mov_b64_e32 v[2:3], s[82:83]
	flat_load_dwordx2 v[16:17], v[2:3]
	v_mov_b64_e32 v[2:3], s[78:79]
	v_mov_b64_e32 v[4:5], s[80:81]
	flat_store_dwordx2 v[2:3], v[4:5]
	v_mov_b64_e32 v[2:3], s[78:79]
	flat_load_dwordx2 v[14:15], v[2:3]
	v_mov_b64_e32 v[2:3], s[74:75]
	v_mov_b64_e32 v[4:5], s[76:77]
	flat_store_dwordx2 v[2:3], v[4:5]
	v_mov_b64_e32 v[2:3], s[74:75]
	flat_load_dwordx2 v[12:13], v[2:3]
	v_mov_b64_e32 v[2:3], s[70:71]
	v_mov_b64_e32 v[4:5], s[72:73]
	flat_store_dwordx2 v[2:3], v[4:5]
	v_mov_b64_e32 v[2:3], s[70:71]
	flat_load_dwordx2 v[10:11], v[2:3]
	v_mov_b64_e32 v[2:3], s[66:67]
	v_mov_b64_e32 v[4:5], s[68:69]
	flat_store_dwordx2 v[2:3], v[4:5]
	v_mov_b64_e32 v[2:3], s[66:67]
	flat_load_dwordx2 v[8:9], v[2:3]
	v_mov_b64_e32 v[2:3], s[62:63]
	v_mov_b64_e32 v[4:5], s[64:65]
	flat_store_dwordx2 v[2:3], v[4:5]
	v_mov_b64_e32 v[2:3], s[62:63]
	flat_load_dwordx2 v[6:7], v[2:3]
	v_mov_b64_e32 v[2:3], s[58:59]
	v_mov_b64_e32 v[4:5], s[60:61]
	flat_store_dwordx2 v[2:3], v[4:5]
	v_mov_b64_e32 v[2:3], s[58:59]
	flat_load_dwordx2 v[2:3], v[2:3]
	v_mov_b64_e32 v[4:5], s[18:19]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dwordx2 v[4:5], v[18:19]
	v_mov_b64_e32 v[4:5], s[56:57]
	flat_store_dwordx2 v[4:5], v[16:17]
	v_mov_b64_e32 v[4:5], s[54:55]
	flat_store_dwordx2 v[4:5], v[14:15]
	v_mov_b64_e32 v[4:5], s[52:53]
	flat_store_dwordx2 v[4:5], v[12:13]
	v_mov_b64_e32 v[4:5], s[50:51]
	flat_store_dwordx2 v[4:5], v[10:11]
	v_mov_b64_e32 v[4:5], s[48:49]
	v_mov_b32_e32 v10, s46
	flat_store_dword v[4:5], v10
	v_mov_b64_e32 v[4:5], s[44:45]
	v_mov_b64_e32 v[10:11], s[42:43]
	flat_store_dwordx2 v[4:5], v[10:11]
	v_mov_b64_e32 v[4:5], s[40:41]
	v_mov_b64_e32 v[10:11], s[38:39]
	flat_store_dwordx2 v[4:5], v[10:11]
	v_mov_b64_e32 v[4:5], s[36:37]
	v_mov_b64_e32 v[10:11], s[34:35]
	flat_store_dwordx2 v[4:5], v[10:11]
	v_mov_b64_e32 v[4:5], s[30:31]
	v_mov_b64_e32 v[10:11], s[28:29]
	flat_store_dwordx2 v[4:5], v[10:11]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b32_e32 v10, s15
	flat_store_dword v[4:5], v10
	v_mov_b64_e32 v[4:5], s[26:27]
	flat_store_dwordx2 v[4:5], v[8:9]
	v_mov_b64_e32 v[4:5], s[24:25]
	flat_store_dwordx2 v[4:5], v[6:7]
	v_mov_b64_e32 v[4:5], s[22:23]
	v_mov_b32_e32 v6, s9
	flat_store_dword v[4:5], v6
	v_mov_b64_e32 v[4:5], s[20:21]
	v_mov_b32_e32 v6, s8
	flat_store_dword v[4:5], v6
	v_mov_b64_e32 v[4:5], s[6:7]
	v_mov_b32_e32 v6, s1
	flat_store_dword v[4:5], v6
	v_mov_b64_e32 v[4:5], s[2:3]
	v_mov_b32_e32 v6, s0
	flat_store_dword v[4:5], v6
	flat_store_dwordx2 v[0:1], v[2:3]
.Ltmp466:
	.loc	56 258 116 prologue_end         ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:258:116
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_group_id@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_group_id@rel32@hi+12
	v_mov_b32_e32 v0, 0
	v_accvgpr_write_b32 a35, v0             ;  Reload Reuse
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s14, v42, 24
	v_readlane_b32 s15, v42, 25
	v_readlane_b32 s12, v42, 50
	v_readlane_b32 s13, v42, 51
	v_readlane_b32 s6, v42, 14
	v_readlane_b32 s8, v42, 26
	v_readlane_b32 s9, v42, 27
	v_readlane_b32 s2, v42, 54
	v_readlane_b32 s3, v42, 55
	v_readlane_b32 s0, v42, 58
	v_readlane_b32 s1, v42, 59
	v_mov_b32_e32 v2, v0
	v_accvgpr_read_b32 v0, a35              ;  Reload Reuse
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v1
.Ltmp467:
	.loc	55 65 29                        ; csrc/cache_kernels_fused.hip:65:29
	v_mov_b32_e32 v1, v3
	s_mov_b64 s[20:21], 0xffffffff
	s_mov_b32 s7, s21
	v_and_b32_e64 v1, v1, s7
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	s_mov_b32 s7, s20
	v_and_b32_e64 v4, v2, s7
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v1
	.loc	55 65 17 is_stmt 0              ; csrc/cache_kernels_fused.hip:65:17
	v_mov_b64_e32 v[2:3], s[16:17]
	flat_store_dwordx2 v[2:3], v[4:5]
	.loc	55 66 23 is_stmt 1              ; csrc/cache_kernels_fused.hip:66:23
	v_mov_b64_e32 v[2:3], s[18:19]
	flat_load_dwordx2 v[4:5], v[2:3]
	.loc	55 66 33 is_stmt 0              ; csrc/cache_kernels_fused.hip:66:33
	v_mov_b64_e32 v[2:3], s[16:17]
	flat_load_dwordx2 v[2:3], v[2:3]
	s_mov_b32 s7, 3
	.loc	55 66 23                        ; csrc/cache_kernels_fused.hip:66:23
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshl_add_u64 v[2:3], v[2:3], s7, v[4:5]
	flat_load_dwordx2 v[4:5], v[2:3]
	.loc	55 66 17                        ; csrc/cache_kernels_fused.hip:66:17
	v_mov_b64_e32 v[2:3], s[12:13]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dwordx2 v[2:3], v[4:5]
	.loc	55 68 29 is_stmt 1              ; csrc/cache_kernels_fused.hip:68:29
	v_mov_b64_e32 v[2:3], s[14:15]
	flat_load_dwordx2 v[2:3], v[2:3]
	.loc	55 68 50 is_stmt 0              ; csrc/cache_kernels_fused.hip:68:50
	v_mov_b64_e32 v[4:5], s[12:13]
	flat_load_dwordx2 v[8:9], v[4:5]
	.loc	55 68 56                        ; csrc/cache_kernels_fused.hip:68:56
	v_mov_b64_e32 v[4:5], s[8:9]
	flat_load_dword v6, v[4:5]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v1, 31, v6
	v_mov_b32_e32 v10, v6
	v_mov_b32_e32 v11, v1
	.loc	55 68 54                        ; csrc/cache_kernels_fused.hip:68:54
	v_lshrrev_b64 v[4:5], s6, v[8:9]
	v_mov_b32_e32 v1, v4
	v_mul_lo_u32 v5, v1, v6
	v_lshrrev_b64 v[10:11], s6, v[10:11]
	v_mov_b32_e32 v4, v10
	v_mov_b32_e32 v1, v8
	v_mul_lo_u32 v4, v1, v4
	v_mad_u64_u32 v[6:7], s[6:7], v1, v6, 0
	v_mov_b32_e32 v1, v7
	v_add3_u32 v4, v1, v4, v5
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v1, s6
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v1
                                        ; kill: def $vgpr6 killed $vgpr6 killed $vgpr6_vgpr7 killed $exec
	s_mov_b32 s6, 0
                                        ; implicit-def: $sgpr6
	v_mov_b32_e32 v1, 0
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v1
	s_mov_b32 s6, 33
	.loc	55 68 48                        ; csrc/cache_kernels_fused.hip:68:48
	v_lshlrev_b64 v[4:5], s6, v[4:5]
	v_mov_b32_e32 v1, v5
	s_mov_b32 s6, 1
	v_lshlrev_b64 v[6:7], s6, v[6:7]
	v_mov_b32_e32 v8, v7
	v_or_b32_e64 v1, v1, v8
                                        ; kill: def $vgpr4 killed $vgpr4 killed $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v6
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v1
	v_lshl_add_u64 v[4:5], v[2:3], 0, v[4:5]
	.loc	55 68 15                        ; csrc/cache_kernels_fused.hip:68:15
	v_mov_b64_e32 v[2:3], s[10:11]
	flat_store_dwordx2 v[2:3], v[4:5]
	.loc	55 70 25 is_stmt 1              ; csrc/cache_kernels_fused.hip:70:25
	v_mov_b64_e32 v[2:3], s[8:9]
	flat_load_dword v1, v[2:3]
	s_mov_b32 s7, 31
	.loc	55 70 33 is_stmt 0              ; csrc/cache_kernels_fused.hip:70:33
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshrrev_b32_e64 v2, s7, v1
	v_add_u32_e64 v1, v1, v2
	v_ashrrev_i32_e64 v1, s6, v1
	.loc	55 70 13                        ; csrc/cache_kernels_fused.hip:70:13
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_store_dword v[2:3], v1
	.loc	55 73 18 is_stmt 1              ; csrc/cache_kernels_fused.hip:73:18
	v_mov_b64_e32 v[2:3], s[4:5]
	flat_load_dword v1, v[2:3]
	.loc	55 73 32 is_stmt 0              ; csrc/cache_kernels_fused.hip:73:32
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v2, v[2:3]
	.loc	55 73 30                        ; csrc/cache_kernels_fused.hip:73:30
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mul_lo_u32 v1, v1, v2
	.loc	55 73 13                        ; csrc/cache_kernels_fused.hip:73:13
	v_mov_b64_e32 v[2:3], s[0:1]
	flat_store_dword v[2:3], v1
.Ltmp468:
	.loc	56 253 117 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:253:117
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_local_id@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_local_id@rel32@hi+12
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s2, v42, 62
	v_readlane_b32 s3, v42, 63
	v_readlane_b32 s0, v42, 11
	v_readlane_b32 s1, v42, 12
	v_mov_b32_e32 v2, v1
                                        ; implicit-def: $sgpr4
                                        ; implicit-def: $sgpr4
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, v0
.Ltmp469:
	.loc	55 74 12                        ; csrc/cache_kernels_fused.hip:74:12
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_store_dword v[0:1], v2
                                        ; implicit-def: $sgpr2_sgpr3
	.loc	55 74 8 is_stmt 0               ; csrc/cache_kernels_fused.hip:74:8
	v_writelane_b32 v43, s0, 10
	s_nop 1
	v_writelane_b32 v43, s1, 11
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
.LBB86_1:                               ; =>This Inner Loop Header: Depth=1
	.loc	55 0 8                          ; csrc/cache_kernels_fused.hip:0:8
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s2, v41, 60
	v_readlane_b32 s3, v41, 61
	v_readlane_b32 s4, v42, 0
	v_readlane_b32 s5, v42, 1
	v_readlane_b32 s0, v43, 12
	v_readlane_b32 s1, v43, 13
	v_readlane_b32 s6, v43, 10
	v_readlane_b32 s7, v43, 11
	s_nop 0
	v_writelane_b32 v43, s6, 14
	s_nop 1
	v_writelane_b32 v43, s7, 15
.Ltmp470:
	.loc	55 74 29 is_stmt 1              ; csrc/cache_kernels_fused.hip:74:29
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dword v0, v[0:1]
	.loc	55 74 33 is_stmt 0              ; csrc/cache_kernels_fused.hip:74:33
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v1, v[2:3]
	.loc	55 74 31                        ; csrc/cache_kernels_fused.hip:74:31
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_i32_e64 s[2:3], v0, v1
	s_mov_b64 s[4:5], -1
	s_or_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v43, s0, 16
	s_nop 1
	v_writelane_b32 v43, s1, 17
.Ltmp471:
	.loc	55 74 3                         ; csrc/cache_kernels_fused.hip:74:3
	v_writelane_b32 v43, s0, 18
	s_nop 1
	v_writelane_b32 v43, s1, 19
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v43, s0, 20
	s_nop 1
	v_writelane_b32 v43, s1, 21
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB86_3
; %bb.2:                                ;   in Loop: Header=BB86_1 Depth=1
.Ltmp472:
	.loc	55 75 20 is_stmt 1              ; csrc/cache_kernels_fused.hip:75:20
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s24, v43, 14
	v_readlane_b32 s25, v43, 15
	v_readlane_b32 s26, v43, 10
	v_readlane_b32 s27, v43, 11
	v_readlane_b32 s28, v43, 12
	v_readlane_b32 s29, v43, 13
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s10, v41, 3
	v_readlane_b32 s11, v41, 4
	v_readlane_b32 s6, v41, 7
	v_readlane_b32 s7, v41, 8
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s8, v43, 8
	v_readlane_b32 s9, v43, 9
	v_readlane_b32 s22, v43, 16
	v_readlane_b32 s23, v43, 17
	v_readlane_b32 s20, v43, 6
	v_readlane_b32 s21, v43, 7
	v_readlane_b32 s16, v43, 18
	v_readlane_b32 s17, v43, 19
	v_readlane_b32 s0, v43, 22
	v_readlane_b32 s1, v43, 23
	v_readlane_b32 s18, v41, 5
	v_readlane_b32 s19, v41, 6
	v_readlane_b32 s30, v43, 4
	v_readlane_b32 s31, v43, 5
	v_readlane_b32 s34, v41, 56
	v_readlane_b32 s35, v41, 57
	v_readlane_b32 s38, v41, 30
	v_readlane_b32 s39, v41, 31
	v_readlane_b32 s40, v43, 2
	v_readlane_b32 s41, v43, 3
	v_readlane_b32 s36, v41, 28
	v_readlane_b32 s37, v41, 29
	v_readlane_b32 s42, v41, 48
	v_readlane_b32 s43, v41, 49
	v_readlane_b32 s44, v41, 18
	v_readlane_b32 s45, v41, 19
	v_readlane_b32 s46, v41, 52
	v_readlane_b32 s47, v41, 53
	v_readlane_b32 s48, v43, 0
	v_readlane_b32 s49, v43, 1
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	v_mov_b64_e32 v[0:1], s[48:49]
	flat_load_dword v3, v[0:1]
	.loc	55 75 24 is_stmt 0              ; csrc/cache_kernels_fused.hip:75:24
	v_mov_b64_e32 v[0:1], s[34:35]
	flat_load_dword v0, v[0:1]
	s_mov_b32 s2, 31
	.loc	55 75 22                        ; csrc/cache_kernels_fused.hip:75:22
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v2, s2, v0
	v_add_u32_e64 v0, v0, v2
	v_xor_b32_e64 v4, v0, v2
	s_mov_b32 s15, 0
	v_sub_u32_e64 v1, s15, v4
	v_cvt_f32_u32_e32 v0, v4
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_mul_lo_u32 v1, v1, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e64 v0, v0, v1
	v_ashrrev_i32_e64 v1, s2, v3
	v_add_u32_e64 v3, v3, v1
	v_xor_b32_e64 v3, v3, v1
	v_mul_hi_u32 v0, v3, v0
	v_mul_lo_u32 v5, v0, v4
	v_sub_u32_e64 v3, v3, v5
	v_cmp_ge_u32_e64 s[52:53], v3, v4
	v_sub_u32_e64 v5, v3, v4
	s_nop 0
	v_cndmask_b32_e64 v3, v3, v5, s[52:53]
	v_cmp_ge_u32_e64 s[50:51], v3, v4
	s_mov_b32 s3, 1
	v_writelane_b32 v42, s3, 22
	v_add_u32_e64 v3, v0, s3
	v_cndmask_b32_e64 v0, v0, v3, s[52:53]
	v_add_u32_e64 v3, v0, s3
	v_cndmask_b32_e64 v0, v0, v3, s[50:51]
	v_xor_b32_e64 v1, v1, v2
	v_xor_b32_e64 v0, v0, v1
	v_sub_u32_e64 v2, v0, v1
	.loc	55 75 9                         ; csrc/cache_kernels_fused.hip:75:9
	v_mov_b64_e32 v[0:1], s[40:41]
	flat_store_dword v[0:1], v2
	.loc	55 76 20 is_stmt 1              ; csrc/cache_kernels_fused.hip:76:20
	v_mov_b64_e32 v[0:1], s[48:49]
	flat_load_dword v0, v[0:1]
	.loc	55 76 24 is_stmt 0              ; csrc/cache_kernels_fused.hip:76:24
	v_mov_b64_e32 v[2:3], s[34:35]
	flat_load_dword v1, v[2:3]
	.loc	55 76 22                        ; csrc/cache_kernels_fused.hip:76:22
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v2, s2, v1
	v_add_u32_e64 v1, v1, v2
	v_xor_b32_e64 v2, v1, v2
	v_sub_u32_e64 v3, s15, v2
	v_cvt_f32_u32_e32 v1, v2
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v3, v3, v1
	v_mul_hi_u32 v3, v1, v3
	v_add_u32_e64 v3, v1, v3
	v_ashrrev_i32_e64 v1, s2, v0
	v_add_u32_e64 v0, v0, v1
	v_xor_b32_e64 v0, v0, v1
	v_mul_hi_u32 v3, v0, v3
	v_mul_lo_u32 v3, v3, v2
	v_sub_u32_e64 v0, v0, v3
	v_cmp_ge_u32_e64 s[48:49], v0, v2
	v_sub_u32_e64 v3, v0, v2
	s_nop 0
	v_cndmask_b32_e64 v0, v0, v3, s[48:49]
	v_cmp_ge_u32_e64 s[48:49], v0, v2
	v_sub_u32_e64 v2, v0, v2
	s_nop 0
	v_cndmask_b32_e64 v0, v0, v2, s[48:49]
	v_xor_b32_e64 v0, v0, v1
	v_sub_u32_e64 v2, v0, v1
	.loc	55 76 9                         ; csrc/cache_kernels_fused.hip:76:9
	v_mov_b64_e32 v[0:1], s[30:31]
	flat_store_dword v[0:1], v2
	.loc	55 80 18 is_stmt 1              ; csrc/cache_kernels_fused.hip:80:18
	v_mov_b64_e32 v[0:1], s[46:47]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 80 32 is_stmt 0              ; csrc/cache_kernels_fused.hip:80:32
	v_mov_b64_e32 v[0:1], s[30:31]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 80 30                        ; csrc/cache_kernels_fused.hip:80:30
	v_lshl_add_u64 v[0:1], v[0:1], s3, v[2:3]
	.loc	55 80 16                        ; csrc/cache_kernels_fused.hip:80:16
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[20:21]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 81 18 is_stmt 1              ; csrc/cache_kernels_fused.hip:81:18
	v_mov_b64_e32 v[0:1], s[46:47]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 81 32 is_stmt 0              ; csrc/cache_kernels_fused.hip:81:32
	v_mov_b64_e32 v[0:1], s[30:31]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 81 30                        ; csrc/cache_kernels_fused.hip:81:30
	v_lshl_add_u64 v[2:3], v[0:1], s3, v[2:3]
	.loc	55 81 43                        ; csrc/cache_kernels_fused.hip:81:43
	v_mov_b64_e32 v[0:1], s[34:35]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 81 41                        ; csrc/cache_kernels_fused.hip:81:41
	v_lshl_add_u64 v[0:1], v[0:1], s3, v[2:3]
	.loc	55 81 16                        ; csrc/cache_kernels_fused.hip:81:16
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[8:9]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 84 9 is_stmt 1               ; csrc/cache_kernels_fused.hip:84:9
	v_mov_b64_e32 v[0:1], s[44:45]
	flat_load_dwordx2 v[0:1], v[0:1]
	.loc	55 84 16 is_stmt 0              ; csrc/cache_kernels_fused.hip:84:16
	v_mov_b64_e32 v[2:3], s[42:43]
	flat_load_dwordx2 v[8:9], v[2:3]
	.loc	55 84 28                        ; csrc/cache_kernels_fused.hip:84:28
	v_mov_b64_e32 v[2:3], s[36:37]
	flat_load_dwordx2 v[2:3], v[2:3]
	s_mov_b32 s2, 32
	.loc	55 84 26                        ; csrc/cache_kernels_fused.hip:84:26
	v_writelane_b32 v42, s2, 23
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshrrev_b64 v[4:5], s2, v[8:9]
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v2
	v_mul_lo_u32 v6, v5, v4
	v_lshrrev_b64 v[2:3], s2, v[2:3]
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v8
	v_mul_lo_u32 v3, v2, v3
	v_mad_u64_u32 v[4:5], s[36:37], v2, v4, 0
	v_mov_b32_e32 v2, v5
	v_add3_u32 v2, v2, v3, v6
                                        ; implicit-def: $sgpr15
                                        ; implicit-def: $sgpr36
                                        ; implicit-def: $sgpr36
	v_mov_b32_e32 v6, s15
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v4
	s_mov_b32 s36, 0
                                        ; implicit-def: $sgpr15
	v_mov_b32_e32 v4, s36
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v4
	s_mov_b32 s15, 33
	.loc	55 84 14                        ; csrc/cache_kernels_fused.hip:84:14
	v_lshlrev_b64 v[2:3], s15, v[2:3]
	v_mov_b32_e32 v4, v3
	v_lshlrev_b64 v[6:7], s3, v[6:7]
	v_mov_b32_e32 v5, v7
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	.loc	55 84 48                        ; csrc/cache_kernels_fused.hip:84:48
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dword v2, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v3, 31, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v3
	.loc	55 84 59                        ; csrc/cache_kernels_fused.hip:84:59
	v_mov_b64_e32 v[6:7], s[38:39]
	flat_load_dwordx2 v[6:7], v[6:7]
	.loc	55 84 57                        ; csrc/cache_kernels_fused.hip:84:57
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshrrev_b64 v[8:9], s2, v[6:7]
	v_mov_b32_e32 v3, v8
	v_mul_lo_u32 v3, v2, v3
	v_lshrrev_b64 v[4:5], s2, v[4:5]
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v6
	v_mul_lo_u32 v6, v5, v4
	v_mad_u64_u32 v[4:5], s[38:39], v2, v4, 0
	v_mov_b32_e32 v2, v5
	v_add3_u32 v2, v2, v3, v6
                                        ; implicit-def: $sgpr37
                                        ; implicit-def: $sgpr38
                                        ; implicit-def: $sgpr38
	v_mov_b32_e32 v6, s37
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v4
                                        ; implicit-def: $sgpr37
	v_mov_b32_e32 v4, s36
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v4
	.loc	55 84 46                        ; csrc/cache_kernels_fused.hip:84:46
	v_lshlrev_b64 v[2:3], s15, v[2:3]
	v_mov_b32_e32 v4, v3
	v_lshlrev_b64 v[6:7], s3, v[6:7]
	v_mov_b32_e32 v5, v7
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[2:3]
	.loc	55 83 11 is_stmt 1              ; csrc/cache_kernels_fused.hip:83:11
	v_mov_b64_e32 v[0:1], s[26:27]
	flat_store_dwordx2 v[0:1], v[2:3]
.Ltmp473:
	.loc	55 89 20                        ; csrc/cache_kernels_fused.hip:89:20
	v_mov_b64_e32 v[0:1], s[30:31]
	flat_load_dword v2, v[0:1]
	.loc	55 89 18 is_stmt 0              ; csrc/cache_kernels_fused.hip:89:18
	v_mov_b64_e32 v[0:1], s[28:29]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dword v[0:1], v2
	.loc	55 90 20 is_stmt 1              ; csrc/cache_kernels_fused.hip:90:20
	v_mov_b64_e32 v[0:1], s[34:35]
	flat_load_dword v0, v[0:1]
	.loc	55 90 32 is_stmt 0              ; csrc/cache_kernels_fused.hip:90:32
	v_mov_b64_e32 v[2:3], s[30:31]
	flat_load_dword v1, v[2:3]
	.loc	55 90 30                        ; csrc/cache_kernels_fused.hip:90:30
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e64 v2, v0, v1
	.loc	55 90 18                        ; csrc/cache_kernels_fused.hip:90:18
	v_mov_b64_e32 v[0:1], s[24:25]
	flat_store_dword v[0:1], v2
.Ltmp474:
	.loc	55 97 18 is_stmt 1              ; csrc/cache_kernels_fused.hip:97:18
	v_mov_b64_e32 v[0:1], s[26:27]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 97 32 is_stmt 0              ; csrc/cache_kernels_fused.hip:97:32
	v_mov_b64_e32 v[0:1], s[28:29]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 97 18                        ; csrc/cache_kernels_fused.hip:97:18
	v_lshl_add_u64 v[0:1], v[0:1], s3, v[2:3]
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[22:23]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 98 18 is_stmt 1              ; csrc/cache_kernels_fused.hip:98:18
	v_mov_b64_e32 v[0:1], s[26:27]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 98 32 is_stmt 0              ; csrc/cache_kernels_fused.hip:98:32
	v_mov_b64_e32 v[0:1], s[24:25]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 98 18                        ; csrc/cache_kernels_fused.hip:98:18
	v_lshl_add_u64 v[0:1], v[0:1], s3, v[2:3]
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[16:17]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 100 18 is_stmt 1             ; csrc/cache_kernels_fused.hip:100:18
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[22:23]
	flat_store_dwordx2 v[0:1], v[2:3]
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[20:21]
	flat_store_dwordx2 v[0:1], v[2:3] offset:8
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[16:17]
	flat_store_dwordx2 v[0:1], v[2:3] offset:16
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[8:9]
	flat_store_dwordx2 v[0:1], v[2:3] offset:24
	s_mov_b64 s[16:17], 0x80
	s_mov_b32 s8, s18
	s_mov_b32 s3, s19
	s_mov_b32 s15, s16
	s_mov_b32 s9, s17
	s_add_u32 s8, s8, s15
	s_addc_u32 s3, s3, s9
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s3
	v_writelane_b32 v42, s8, 24
	s_nop 1
	v_writelane_b32 v42, s9, 25
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v42            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_lshr_b64 s[2:3], s[0:1], s2
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	s_mov_b32 s3, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv@rel32@lo+4
	s_addc_u32 s1, s1, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv@rel32@hi+12
                                        ; implicit-def: $sgpr15
	v_mov_b32_e32 v0, s3
	v_mov_b32_e32 v1, s2
	s_swappc_b64 s[30:31], s[0:1]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	v_readlane_b32 s22, v43, 18
	v_readlane_b32 s23, v43, 19
	v_readlane_b32 s20, v43, 6
	v_readlane_b32 s21, v43, 7
	v_readlane_b32 s18, v43, 16
	v_readlane_b32 s19, v43, 17
	v_readlane_b32 s16, v43, 8
	v_readlane_b32 s17, v43, 9
	v_readlane_b32 s2, v42, 23
	v_readlane_b32 s0, v43, 26
	v_readlane_b32 s1, v43, 27
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s6, v41, 7
	v_readlane_b32 s7, v41, 8
	v_readlane_b32 s8, v42, 24
	v_readlane_b32 s9, v42, 25
	v_readlane_b32 s10, v41, 3
	v_readlane_b32 s11, v41, 4
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s24, v43, 20
	v_readlane_b32 s25, v43, 21
	v_mov_b32_e32 v2, v0
	s_nop 0
	v_mov_b64_e32 v[0:1], s[24:25]
	flat_store_short v[0:1], v2
	.loc	55 107 18                       ; csrc/cache_kernels_fused.hip:107:18
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[22:23]
	flat_store_dwordx2 v[0:1], v[2:3]
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[20:21]
	flat_store_dwordx2 v[0:1], v[2:3] offset:8
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[18:19]
	flat_store_dwordx2 v[0:1], v[2:3] offset:16
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[16:17]
	flat_store_dwordx2 v[0:1], v[2:3] offset:24
	s_lshr_b64 s[2:3], s[0:1], s2
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	s_mov_b32 s3, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv@rel32@lo+4
	s_addc_u32 s1, s1, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv@rel32@hi+12
                                        ; implicit-def: $sgpr15
	v_mov_b32_e32 v0, s3
	v_mov_b32_e32 v1, s2
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s10, v43, 12
	v_readlane_b32 s11, v43, 13
	v_readlane_b32 s8, v43, 20
	v_readlane_b32 s9, v43, 21
	v_readlane_b32 s6, v43, 10
	v_readlane_b32 s7, v43, 11
	v_readlane_b32 s4, v43, 14
	v_readlane_b32 s5, v43, 15
	v_readlane_b32 s2, v42, 22
	v_readlane_b32 s0, v43, 24
	v_readlane_b32 s1, v43, 25
	v_mov_b32_e32 v2, v0
	s_nop 0
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_short v[0:1], v2
	.loc	55 115 5                        ; csrc/cache_kernels_fused.hip:115:5
	v_mov_b64_e32 v[0:1], s[6:7]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 115 19 is_stmt 0             ; csrc/cache_kernels_fused.hip:115:19
	v_mov_b64_e32 v[0:1], s[10:11]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 115 5                        ; csrc/cache_kernels_fused.hip:115:5
	v_lshl_add_u64 v[0:1], v[0:1], s2, v[2:3]
	.loc	55 115 31                       ; csrc/cache_kernels_fused.hip:115:31
	v_mov_b64_e32 v[2:3], s[8:9]
	flat_load_ushort v2, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 116 5 is_stmt 1              ; csrc/cache_kernels_fused.hip:116:5
	v_mov_b64_e32 v[0:1], s[6:7]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 116 19 is_stmt 0             ; csrc/cache_kernels_fused.hip:116:19
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 116 5                        ; csrc/cache_kernels_fused.hip:116:5
	v_lshl_add_u64 v[0:1], v[0:1], s2, v[2:3]
	.loc	55 116 31                       ; csrc/cache_kernels_fused.hip:116:31
	v_mov_b64_e32 v[2:3], s[0:1]
	flat_load_ushort v2, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 117 3 is_stmt 1              ; csrc/cache_kernels_fused.hip:117:3
	s_branch .LBB86_4
.Ltmp475:
.LBB86_3:                               ; %Flow24
                                        ;   in Loop: Header=BB86_1 Depth=1
	.loc	55 0 3 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:3
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s0, v43, 20
	v_readlane_b32 s1, v43, 21
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s4, v43, 14
	v_readlane_b32 s5, v43, 15
	v_readlane_b32 s2, v43, 18
	v_readlane_b32 s3, v43, 19
	s_mov_b64 s[0:1], s[2:3]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_writelane_b32 v43, s2, 12
	s_nop 1
	v_writelane_b32 v43, s3, 13
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v43, s2, 10
	s_nop 1
	v_writelane_b32 v43, s3, 11
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v43, s2, 26
	s_nop 1
	v_writelane_b32 v43, s3, 27
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB86_1
	s_branch .LBB86_5
.LBB86_4:                               ;   in Loop: Header=BB86_1 Depth=1
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s0, v41, 5
	v_readlane_b32 s1, v41, 6
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_mov_b64 s[6:7], 0x80
.Ltmp476:
	.loc	56 263 116 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:263:116
	s_mov_b32 s2, s0
	s_mov_b32 s0, s1
	s_mov_b32 s3, s6
	s_mov_b32 s1, s7
	s_add_u32 s8, s2, s3
	s_addc_u32 s0, s0, s1
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_local_size@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_local_size@rel32@hi+12
	v_mov_b32_e32 v0, 0
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s2, v42, 0
	v_readlane_b32 s3, v42, 1
	v_readlane_b32 s0, v43, 16
	v_readlane_b32 s1, v43, 17
	v_mov_b32_e32 v2, v1
                                        ; implicit-def: $sgpr4
                                        ; implicit-def: $sgpr4
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v1, v0
.Ltmp477:
	.loc	55 74 39                        ; csrc/cache_kernels_fused.hip:74:39
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v0, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e64 v2, v0, v1
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_store_dword v[0:1], v2
	s_mov_b64 s[2:3], 0
	s_andn2_b64 s[0:1], s[0:1], exec
	.loc	55 74 3 is_stmt 0               ; csrc/cache_kernels_fused.hip:74:3
	v_writelane_b32 v43, s0, 18
	s_nop 1
	v_writelane_b32 v43, s1, 19
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_branch .LBB86_3
.Ltmp478:
.LBB86_5:
	.loc	55 0 3                          ; csrc/cache_kernels_fused.hip:0:3
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s0, v43, 26
	v_readlane_b32 s1, v43, 27
	s_or_b64 exec, exec, s[0:1]
; %bb.6:
	.loc	55 119 28 is_stmt 1             ; csrc/cache_kernels_fused.hip:119:28
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s2, v41, 28
	v_readlane_b32 s3, v41, 29
	v_readlane_b32 s4, v41, 32
	v_readlane_b32 s5, v41, 33
	v_readlane_b32 s20, v42, 46
	v_readlane_b32 s21, v42, 47
	v_readlane_b32 s22, v41, 30
	v_readlane_b32 s23, v41, 31
	v_readlane_b32 s0, v42, 48
	v_readlane_b32 s1, v42, 49
	v_readlane_b32 s6, v42, 38
	v_readlane_b32 s7, v42, 39
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_mov_b64_e32 v[0:1], s[6:7]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 119 50 is_stmt 0             ; csrc/cache_kernels_fused.hip:119:50
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_load_dwordx2 v[0:1], v[0:1]
	s_mov_b32 s0, 3
	.loc	55 119 28                       ; csrc/cache_kernels_fused.hip:119:28
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshl_add_u64 v[0:1], v[0:1], s0, v[2:3]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 119 17                       ; csrc/cache_kernels_fused.hip:119:17
	v_mov_b64_e32 v[0:1], s[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dwordx2 v[0:1], v[2:3]
	.loc	55 120 29 is_stmt 1             ; csrc/cache_kernels_fused.hip:120:29
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[0:1], v[0:1]
	.loc	55 120 40 is_stmt 0             ; csrc/cache_kernels_fused.hip:120:40
	v_mov_b64_e32 v[2:3], s[20:21]
	flat_load_dword v2, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v2
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	s_mov_b64 s[10:11], 0
	.loc	55 120 38                       ; csrc/cache_kernels_fused.hip:120:38
	v_writelane_b32 v43, s10, 28
	s_nop 1
	v_writelane_b32 v43, s11, 29
	v_cmp_lt_i64_e64 s[6:7], v[2:3], s[10:11]
	s_mov_b64 s[0:1], -1
	s_mov_b32 s14, s1
	s_mov_b32 s15, s11
	v_mov_b32_e32 v4, s15
	v_mov_b32_e32 v5, s14
	v_cndmask_b32_e64 v6, v4, v5, s[6:7]
	s_mov_b32 s12, s0
	s_mov_b32 s13, s10
	v_mov_b32_e32 v4, s13
	v_mov_b32_e32 v5, s12
	v_cndmask_b32_e64 v4, v4, v5, s[6:7]
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr6
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v6
	v_mov_b32_e32 v6, v5
	v_lshl_add_u64 v[8:9], v[2:3], 0, v[4:5]
	v_mov_b32_e32 v2, v9
	v_xor_b32_e64 v2, v2, v6
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v3, v8
	v_xor_b32_e64 v8, v3, v5
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v2
	v_mov_b32_e32 v14, v8
	v_cvt_f32_u32_e64 v2, v14
	s_mov_b32 s8, 32
	v_writelane_b32 v43, s8, 30
	v_lshrrev_b64 v[10:11], s8, v[8:9]
	v_mov_b32_e32 v16, v10
	v_cvt_f32_u32_e64 v3, v16
	s_mov_b32 s19, 0x4f800000
	v_fmac_f32_e64 v2, v3, s19
	v_rcp_f32_e64 v2, v2
	s_mov_b32 s18, 0x5f7ffffc
	v_mul_f32_e64 v3, v2, s18
	s_mov_b32 s17, 0x2f800000
	v_mul_f32_e64 v2, v3, s17
	v_trunc_f32_e64 v2, v2
	s_mov_b32 s16, 0xcf800000
	v_fmac_f32_e64 v3, v2, s16
	v_cvt_u32_f32_e64 v3, v3
	s_mov_b32 s6, s10
	v_mov_b32_e32 v4, v8
	s_mov_b32 s9, s11
	v_mov_b32_e32 v7, v9
	v_sub_co_u32_e64 v12, s[6:7], s6, v4
	v_mov_b32_e32 v4, s9
	s_nop 0
	v_subb_co_u32_e64 v4, s[6:7], v4, v7, s[6:7]
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v4
	v_lshrrev_b64 v[8:9], s8, v[12:13]
                                        ; kill: def $vgpr8 killed $vgpr8 killed $vgpr8_vgpr9 killed $exec
	v_mul_lo_u32 v10, v8, v3
	v_cvt_u32_f32_e64 v2, v2
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr6
	v_mov_b32_e32 v18, v3
	v_mov_b32_e32 v19, v2
	v_lshrrev_b64 v[18:19], s8, v[18:19]
	v_mov_b32_e32 v7, v18
	v_mov_b32_e32 v11, v12
	v_mul_lo_u32 v9, v11, v7
	v_mad_u64_u32 v[18:19], s[6:7], v11, v3, 0
	v_mov_b32_e32 v4, v19
	v_add3_u32 v13, v4, v9, v10
	v_mad_u64_u32 v[20:21], s[6:7], v3, v13, 0
	v_mov_b32_e32 v22, v20
	s_mov_b32 s9, 0
	v_writelane_b32 v43, s9, 31
                                        ; implicit-def: $sgpr6
	v_mov_b32_e32 v4, s9
                                        ; kill: def $vgpr22 killed $vgpr22 def $vgpr22_vgpr23 killed $exec
	v_mov_b32_e32 v23, v4
	v_mov_b32_e32 v4, v23
	v_mov_b32_e32 v20, v21
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v9, s6
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v9
	v_lshlrev_b64 v[20:21], s8, v[20:21]
	v_mov_b32_e32 v9, v21
	v_or_b32_e64 v4, v4, v9
	v_mov_b32_e32 v9, v22
	v_mov_b32_e32 v10, v20
	v_or_b32_e64 v20, v9, v10
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v4
	v_mov_b32_e32 v9, v18
	v_mul_hi_u32 v18, v3, v9
                                        ; implicit-def: $sgpr6
	v_mov_b32_e32 v4, s9
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v4
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[20:21]
	v_mov_b32_e32 v10, v18
	v_mov_b32_e32 v4, v19
	v_mad_u64_u32 v[18:19], s[6:7], v7, v9, 0
	v_mov_b32_e32 v20, v18
                                        ; implicit-def: $sgpr6
	v_mov_b32_e32 v9, s9
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v9
	v_mov_b32_e32 v9, v21
	v_mov_b32_e32 v18, v19
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v12, s6
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v12
	v_lshlrev_b64 v[18:19], s8, v[18:19]
	v_mov_b32_e32 v12, v19
	v_or_b32_e64 v9, v9, v12
	v_mov_b32_e32 v12, v20
	v_mov_b32_e32 v15, v18
	v_or_b32_e64 v18, v12, v15
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v9
	v_mov_b32_e32 v12, v18
	v_mov_b32_e32 v9, v19
	v_mad_u64_u32 v[18:19], s[6:7], v7, v13, 0
	v_mov_b32_e32 v7, v19
	s_mov_b32 s6, 0
	v_writelane_b32 v43, s6, 32
	v_add_co_u32_e32 v12, vcc, v10, v12
	s_nop 1
	v_addc_co_u32_e32 v4, vcc, v4, v9, vcc
	v_mov_b32_e32 v9, s6
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v7, v9, vcc
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr24
                                        ; implicit-def: $sgpr24
	v_mov_b32_e32 v7, s7
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v7
	v_lshlrev_b64 v[20:21], s8, v[20:21]
	v_mov_b32_e32 v9, v21
                                        ; kill: def $vgpr18 killed $vgpr18 killed $vgpr18_vgpr19 killed $exec
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v7, s9
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v7
	v_mov_b32_e32 v7, v19
	v_or_b32_e64 v7, v7, v9
	v_mov_b32_e32 v10, v20
	v_mov_b32_e32 v9, v18
	v_or_b32_e64 v18, v9, v10
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v7
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v4
	v_lshrrev_b64 v[12:13], s8, v[12:13]
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[18:19]
	v_mov_b32_e32 v4, v12
	v_add_co_u32_e64 v3, s[24:25], v3, v4
	v_lshrrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v4, v12
	v_addc_co_u32_e64 v2, s[24:25], v2, v4, s[24:25]
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v12, v3
	v_mov_b32_e32 v13, v2
	v_lshrrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v7, v12
	v_mad_u64_u32 v[18:19], s[24:25], v11, v3, 0
	v_mov_b32_e32 v4, v18
	v_mad_u64_u32 v[12:13], s[24:25], v7, v4, 0
	v_mov_b32_e32 v20, v12
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v9, s9
                                        ; kill: def $vgpr20 killed $vgpr20 def $vgpr20_vgpr21 killed $exec
	v_mov_b32_e32 v21, v9
	v_mov_b32_e32 v9, v21
	v_mov_b32_e32 v12, v13
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr24
                                        ; implicit-def: $sgpr24
	v_mov_b32_e32 v10, s7
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v10
	v_lshlrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v10, v13
	v_or_b32_e64 v9, v9, v10
	v_mov_b32_e32 v10, v20
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	v_or_b32_e64 v12, v10, v12
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v9
	v_mov_b32_e32 v10, v12
	v_mov_b32_e32 v9, v13
	v_mul_lo_u32 v11, v11, v7
	v_mul_lo_u32 v12, v8, v3
	v_mov_b32_e32 v8, v19
	v_add3_u32 v11, v8, v11, v12
	v_mad_u64_u32 v[18:19], s[24:25], v3, v11, 0
	v_mov_b32_e32 v12, v18
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v8, s9
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v8
	v_mov_b32_e32 v8, v13
	v_mov_b32_e32 v18, v19
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr24
                                        ; implicit-def: $sgpr24
	v_mov_b32_e32 v15, s7
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v15
	v_lshlrev_b64 v[18:19], s8, v[18:19]
	v_mov_b32_e32 v15, v19
	v_or_b32_e64 v8, v8, v15
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v18
	v_or_b32_e64 v18, v12, v13
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v8
	v_mul_hi_u32 v12, v3, v4
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v4, s9
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v4
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[18:19]
	v_mov_b32_e32 v8, v12
	v_mov_b32_e32 v4, v13
	v_mad_u64_u32 v[12:13], s[24:25], v7, v11, 0
	v_mov_b32_e32 v7, v13
	v_add_co_u32_e32 v8, vcc, v8, v10
	s_nop 1
	v_addc_co_u32_e32 v4, vcc, v4, v9, vcc
	v_mov_b32_e32 v9, s6
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v7, v9, vcc
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr24
                                        ; implicit-def: $sgpr24
	v_mov_b32_e32 v7, s7
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v7
	v_lshlrev_b64 v[10:11], s8, v[10:11]
	v_mov_b32_e32 v9, v11
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v7, s9
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v7
	v_mov_b32_e32 v7, v13
	v_or_b32_e64 v7, v7, v9
                                        ; kill: def $vgpr10 killed $vgpr10 killed $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v9, v12
	v_or_b32_e64 v10, v9, v10
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v7
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v4
	v_lshrrev_b64 v[8:9], s8, v[8:9]
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_mov_b32_e32 v4, v10
	v_add_co_u32_e64 v9, s[24:25], v3, v4
	v_lshrrev_b64 v[10:11], s8, v[10:11]
	v_mov_b32_e32 v3, v10
	v_addc_co_u32_e64 v4, s[24:25], v2, v3, s[24:25]
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v2, v9
	v_mov_b32_e32 v3, v4
	v_lshrrev_b64 v[2:3], s8, v[2:3]
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_cmp_lt_i64_e64 s[24:25], v[0:1], s[10:11]
	v_mov_b32_e32 v3, s15
	v_mov_b32_e32 v4, s14
	v_cndmask_b32_e64 v3, v3, v4, s[24:25]
	v_mov_b32_e32 v4, s13
	v_mov_b32_e32 v7, s12
	v_cndmask_b32_e64 v12, v4, v7, s[24:25]
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v3
	v_mov_b32_e32 v3, v13
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[12:13]
	v_mov_b32_e32 v0, v11
	v_xor_b32_e64 v0, v0, v3
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v1, v10
	v_xor_b32_e64 v10, v1, v4
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v7, v10
	v_mad_u64_u32 v[12:13], s[24:25], v7, v2, 0
	v_mov_b32_e32 v18, v12
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v0, s9
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v0, v19
	v_mov_b32_e32 v12, v13
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr24
                                        ; implicit-def: $sgpr24
	v_mov_b32_e32 v1, s7
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v1
	v_lshlrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v1, v13
	v_or_b32_e64 v0, v0, v1
	v_mov_b32_e32 v1, v18
	v_mov_b32_e32 v8, v12
	v_or_b32_e64 v12, v1, v8
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v0
	v_mul_hi_u32 v0, v7, v9
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v8, s9
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v8
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[12:13]
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v0, v1
	v_lshrrev_b64 v[10:11], s8, v[10:11]
	v_mov_b32_e32 v1, v10
	v_mad_u64_u32 v[12:13], s[24:25], v1, v9, 0
	v_mov_b32_e32 v10, v12
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v9, s9
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v9
	v_mov_b32_e32 v9, v11
	v_mov_b32_e32 v12, v13
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr24
                                        ; implicit-def: $sgpr24
	v_mov_b32_e32 v15, s7
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v15
	v_lshlrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v15, v13
	v_or_b32_e64 v9, v9, v15
                                        ; kill: def $vgpr10 killed $vgpr10 killed $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v12
	v_or_b32_e64 v12, v10, v11
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v9
	v_mov_b32_e32 v10, v12
	v_mov_b32_e32 v9, v13
	v_mad_u64_u32 v[12:13], s[24:25], v1, v2, 0
	v_mov_b32_e32 v2, v13
	v_add_co_u32_e32 v8, vcc, v8, v10
	s_nop 1
	v_addc_co_u32_e32 v0, vcc, v0, v9, vcc
	v_mov_b32_e32 v9, s6
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v2, v9, vcc
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr24
                                        ; implicit-def: $sgpr24
	v_mov_b32_e32 v2, s7
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v2
	v_lshlrev_b64 v[10:11], s8, v[10:11]
	v_mov_b32_e32 v9, v11
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
                                        ; implicit-def: $sgpr7
	v_mov_b32_e32 v2, s9
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v2
	v_mov_b32_e32 v2, v13
	v_or_b32_e64 v2, v2, v9
                                        ; kill: def $vgpr10 killed $vgpr10 killed $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v9, v12
	v_or_b32_e64 v10, v9, v10
                                        ; kill: def $vgpr10 killed $vgpr10 def $vgpr10_vgpr11 killed $exec
	v_mov_b32_e32 v11, v2
                                        ; implicit-def: $sgpr7
                                        ; implicit-def: $sgpr7
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v0
	v_lshrrev_b64 v[8:9], s8, v[8:9]
	v_lshl_add_u64 v[12:13], v[8:9], 0, v[10:11]
	v_mov_b32_e32 v0, v12
	v_mul_lo_u32 v11, v16, v0
	v_lshrrev_b64 v[8:9], s8, v[12:13]
	v_mov_b32_e32 v2, v8
	v_mul_lo_u32 v10, v14, v2
	v_mad_u64_u32 v[8:9], s[24:25], v14, v0, 0
	v_mov_b32_e32 v2, v9
	v_add3_u32 v15, v2, v10, v11
	v_sub_u32_e64 v2, v1, v15
                                        ; kill: def $vgpr8 killed $vgpr8 killed $vgpr8_vgpr9 killed $exec
	v_sub_co_u32_e64 v7, s[24:25], v7, v8
	s_nop 1
	v_subb_co_u32_e64 v2, s[26:27], v2, v16, s[24:25]
	v_sub_co_u32_e64 v8, s[26:27], v7, v14
	v_mov_b32_e32 v9, s6
	s_nop 0
	v_subb_co_u32_e64 v9, s[26:27], v2, v9, s[26:27]
	v_cmp_ge_u32_e64 s[26:27], v9, v16
	s_mov_b32 s7, -1
	v_writelane_b32 v43, s7, 33
	v_mov_b32_e32 v2, s6
	v_mov_b32_e32 v10, s7
	v_cndmask_b32_e64 v2, v2, v10, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], v9, v16
	v_cmp_ge_u32_e64 s[28:29], v8, v14
	v_mov_b32_e32 v8, s6
	v_mov_b32_e32 v9, s7
	v_cndmask_b32_e64 v8, v8, v9, s[28:29]
	v_cndmask_b32_e64 v2, v2, v8, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], v2, s6
	s_mov_b64 s[28:29], 2
	v_lshl_add_u64 v[10:11], v[12:13], 0, s[28:29]
	v_mov_b32_e32 v17, v11
	s_mov_b64 s[28:29], 1
	v_lshl_add_u64 v[8:9], v[12:13], 0, s[28:29]
	v_mov_b32_e32 v2, v9
	v_cndmask_b32_e64 v2, v2, v17, s[26:27]
	v_subb_co_u32_e64 v15, s[24:25], v1, v15, s[24:25]
	v_cmp_ge_u32_e64 s[24:25], v15, v16
	v_mov_b32_e32 v1, s6
	v_mov_b32_e32 v17, s7
	v_cndmask_b32_e64 v1, v1, v17, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], v15, v16
	v_cmp_ge_u32_e64 s[28:29], v7, v14
	v_mov_b32_e32 v7, s6
	v_mov_b32_e32 v14, s7
	v_cndmask_b32_e64 v7, v7, v14, s[28:29]
	v_cndmask_b32_e64 v1, v1, v7, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], v1, s6
	v_mov_b32_e32 v1, v13
	s_nop 0
	v_cndmask_b32_e64 v2, v1, v2, s[24:25]
	v_mov_b32_e32 v7, v10
	v_mov_b32_e32 v1, v8
	v_cndmask_b32_e64 v1, v1, v7, s[26:27]
	v_cndmask_b32_e64 v0, v0, v1, s[24:25]
                                        ; implicit-def: $sgpr24
                                        ; implicit-def: $sgpr24
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, v1
	v_xor_b32_e64 v3, v3, v6
	v_xor_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v3
	v_mov_b32_e32 v3, v5
	v_xor_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr0 killed $vgpr0 killed $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	v_xor_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v4
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v1, v5
	v_sub_co_u32_e64 v2, s[24:25], v2, v3
	s_nop 1
	v_subb_co_u32_e64 v0, s[24:25], v0, v1, s[24:25]
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v0
	.loc	55 120 17                       ; csrc/cache_kernels_fused.hip:120:17
	v_mov_b64_e32 v[0:1], s[22:23]
	flat_store_dwordx2 v[0:1], v[2:3]
	.loc	55 121 29 is_stmt 1             ; csrc/cache_kernels_fused.hip:121:29
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[0:1], v[0:1]
	.loc	55 121 40 is_stmt 0             ; csrc/cache_kernels_fused.hip:121:40
	v_mov_b64_e32 v[2:3], s[20:21]
	flat_load_dword v6, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v2, 31, v6
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v2
	.loc	55 121 38                       ; csrc/cache_kernels_fused.hip:121:38
	v_cmp_lt_i64_e64 s[20:21], v[6:7], s[10:11]
	v_mov_b32_e32 v2, s15
	v_mov_b32_e32 v3, s14
	v_cndmask_b32_e64 v2, v2, v3, s[20:21]
	v_mov_b32_e32 v3, s13
	v_mov_b32_e32 v4, s12
	v_cndmask_b32_e64 v4, v3, v4, s[20:21]
                                        ; implicit-def: $sgpr20
                                        ; implicit-def: $sgpr20
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v3, v5
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[4:5]
	v_mov_b32_e32 v2, v7
	v_xor_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr4 killed $vgpr4 killed $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v3, v6
	v_xor_b32_e64 v6, v3, v4
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v2
	v_mov_b32_e32 v10, v6
	v_cvt_f32_u32_e64 v2, v10
	v_lshrrev_b64 v[4:5], s8, v[6:7]
	v_mov_b32_e32 v11, v4
	v_accvgpr_write_b32 a37, v11            ;  Reload Reuse
	v_cvt_f32_u32_e64 v3, v11
	v_fmac_f32_e64 v2, v3, s19
	v_rcp_f32_e64 v2, v2
	s_nop 0
	v_mul_f32_e64 v3, v2, s18
	v_mul_f32_e64 v2, v3, s17
	v_trunc_f32_e64 v2, v2
	v_fmac_f32_e64 v3, v2, s16
	v_cvt_u32_f32_e64 v3, v3
	s_mov_b32 s16, s10
	v_mov_b32_e32 v4, v6
	s_mov_b32 s18, s11
	v_mov_b32_e32 v5, v7
	v_sub_co_u32_e64 v12, s[16:17], s16, v4
	v_mov_b32_e32 v4, s18
	s_nop 0
	v_subb_co_u32_e64 v4, s[16:17], v4, v5, s[16:17]
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v4
	v_lshrrev_b64 v[4:5], s8, v[12:13]
	v_mov_b32_e32 v6, v4
	v_mul_lo_u32 v8, v6, v3
	v_cvt_u32_f32_e64 v2, v2
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v2
	v_lshrrev_b64 v[4:5], s8, v[4:5]
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v9, v12
	v_mul_lo_u32 v7, v9, v5
	v_mad_u64_u32 v[14:15], s[16:17], v9, v3, 0
	v_mov_b32_e32 v4, v15
	v_add3_u32 v13, v4, v7, v8
	v_mad_u64_u32 v[16:17], s[16:17], v3, v13, 0
	v_mov_b32_e32 v18, v16
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v4, s9
                                        ; kill: def $vgpr18 killed $vgpr18 def $vgpr18_vgpr19 killed $exec
	v_mov_b32_e32 v19, v4
	v_mov_b32_e32 v4, v19
	v_mov_b32_e32 v16, v17
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr17
                                        ; implicit-def: $sgpr17
	v_mov_b32_e32 v7, s16
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v7
	v_lshlrev_b64 v[16:17], s8, v[16:17]
	v_mov_b32_e32 v7, v17
	v_or_b32_e64 v4, v4, v7
	v_mov_b32_e32 v7, v18
	v_mov_b32_e32 v8, v16
	v_or_b32_e64 v16, v7, v8
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v4
	v_mov_b32_e32 v8, v14
	v_mul_hi_u32 v14, v3, v8
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v4, s9
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v4
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mov_b32_e32 v4, v14
	v_mov_b32_e32 v7, v15
	v_mad_u64_u32 v[14:15], s[16:17], v5, v8, 0
	v_mov_b32_e32 v16, v14
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v8, s9
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v8
	v_mov_b32_e32 v8, v17
	v_mov_b32_e32 v14, v15
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr17
                                        ; implicit-def: $sgpr17
	v_mov_b32_e32 v12, s16
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v12
	v_lshlrev_b64 v[14:15], s8, v[14:15]
	v_mov_b32_e32 v12, v15
	v_or_b32_e64 v8, v8, v12
	v_mov_b32_e32 v12, v16
                                        ; kill: def $vgpr14 killed $vgpr14 killed $vgpr14_vgpr15 killed $exec
	v_or_b32_e64 v14, v12, v14
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v8
	v_mov_b32_e32 v12, v14
	v_mov_b32_e32 v8, v15
	v_mad_u64_u32 v[14:15], s[16:17], v5, v13, 0
	v_mov_b32_e32 v5, v15
	v_add_co_u32_e32 v4, vcc, v4, v12
	s_nop 1
	v_addc_co_u32_e32 v7, vcc, v7, v8, vcc
	v_mov_b32_e32 v8, s6
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v5, v8, vcc
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr17
                                        ; implicit-def: $sgpr17
	v_mov_b32_e32 v5, s16
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v5
	v_lshlrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v8, v13
                                        ; kill: def $vgpr14 killed $vgpr14 killed $vgpr14_vgpr15 killed $exec
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v5, s9
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v5
	v_mov_b32_e32 v5, v15
	v_or_b32_e64 v5, v5, v8
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v8, v14
	v_or_b32_e64 v12, v8, v12
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v5
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr16
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v7
	v_lshrrev_b64 v[4:5], s8, v[4:5]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[12:13]
	v_mov_b32_e32 v7, v4
	v_add_co_u32_e64 v3, s[16:17], v3, v7
	v_lshrrev_b64 v[4:5], s8, v[4:5]
                                        ; kill: def $vgpr4 killed $vgpr4 killed $vgpr4_vgpr5 killed $exec
	s_nop 0
	v_addc_co_u32_e64 v2, s[16:17], v2, v4, s[16:17]
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v4, v3
	v_mov_b32_e32 v5, v2
	v_lshrrev_b64 v[4:5], s8, v[4:5]
	v_mov_b32_e32 v5, v4
	v_mad_u64_u32 v[14:15], s[16:17], v9, v3, 0
	v_mov_b32_e32 v4, v14
	v_mad_u64_u32 v[12:13], s[16:17], v5, v4, 0
	v_mov_b32_e32 v16, v12
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v7, s9
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v7
	v_mov_b32_e32 v7, v17
	v_mov_b32_e32 v12, v13
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr17
                                        ; implicit-def: $sgpr17
	v_mov_b32_e32 v8, s16
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v8
	v_lshlrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v8, v13
	v_or_b32_e64 v7, v7, v8
	v_mov_b32_e32 v8, v16
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	v_or_b32_e64 v12, v8, v12
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v7
	v_mov_b32_e32 v8, v12
	v_mov_b32_e32 v7, v13
	v_mul_lo_u32 v9, v9, v5
	v_mul_lo_u32 v12, v6, v3
	v_mov_b32_e32 v6, v15
	v_add3_u32 v9, v6, v9, v12
	v_mad_u64_u32 v[14:15], s[16:17], v3, v9, 0
	v_mov_b32_e32 v12, v14
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v6, s9
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v6
	v_mov_b32_e32 v6, v13
	v_mov_b32_e32 v14, v15
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr17
                                        ; implicit-def: $sgpr17
	v_mov_b32_e32 v16, s16
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v16
	v_lshlrev_b64 v[14:15], s8, v[14:15]
	v_mov_b32_e32 v16, v15
	v_or_b32_e64 v6, v6, v16
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v14
	v_or_b32_e64 v14, v12, v13
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v6
	v_mul_hi_u32 v12, v3, v4
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v4, s9
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v4
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v6, v13
	v_mad_u64_u32 v[12:13], s[16:17], v5, v9, 0
	v_mov_b32_e32 v5, v13
	v_add_co_u32_e32 v4, vcc, v4, v8
	s_nop 1
	v_addc_co_u32_e32 v8, vcc, v6, v7, vcc
	v_mov_b32_e32 v6, s6
	s_nop 0
	v_addc_co_u32_e32 v6, vcc, v5, v6, vcc
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr17
                                        ; implicit-def: $sgpr17
	v_mov_b32_e32 v5, s16
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v5
	v_lshlrev_b64 v[6:7], s8, v[6:7]
	v_mov_b32_e32 v9, v7
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v5, s9
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v5
	v_mov_b32_e32 v5, v13
	v_or_b32_e64 v5, v5, v9
	v_mov_b32_e32 v7, v6
	v_mov_b32_e32 v6, v12
	v_or_b32_e64 v6, v6, v7
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v5
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr16
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v8
	v_lshrrev_b64 v[4:5], s8, v[4:5]
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	v_mov_b32_e32 v6, v4
	v_add_co_u32_e64 v9, s[16:17], v3, v6
	v_lshrrev_b64 v[4:5], s8, v[4:5]
	v_mov_b32_e32 v3, v4
	v_addc_co_u32_e64 v4, s[16:17], v2, v3, s[16:17]
                                        ; implicit-def: $sgpr16
                                        ; implicit-def: $sgpr16
	v_mov_b32_e32 v2, v9
	v_mov_b32_e32 v3, v4
	v_lshrrev_b64 v[2:3], s8, v[2:3]
	v_mov_b32_e32 v7, v2
	v_cmp_lt_i64_e64 s[10:11], v[0:1], s[10:11]
	v_mov_b32_e32 v2, s15
	v_mov_b32_e32 v3, s14
	v_cndmask_b32_e64 v2, v2, v3, s[10:11]
	v_mov_b32_e32 v3, s13
	v_mov_b32_e32 v4, s12
	v_cndmask_b32_e64 v4, v3, v4, s[10:11]
                                        ; implicit-def: $sgpr10
                                        ; implicit-def: $sgpr10
                                        ; kill: def $vgpr4 killed $vgpr4 def $vgpr4_vgpr5 killed $exec
	v_mov_b32_e32 v5, v2
	v_mov_b32_e32 v2, v5
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[4:5]
	v_mov_b32_e32 v0, v13
	v_xor_b32_e64 v0, v0, v2
	v_mov_b32_e32 v1, v4
	v_mov_b32_e32 v3, v12
	v_xor_b32_e64 v12, v3, v1
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v3, v12
	v_mad_u64_u32 v[14:15], s[10:11], v3, v7, 0
	v_mov_b32_e32 v16, v14
                                        ; implicit-def: $sgpr10
	v_mov_b32_e32 v0, s9
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v0, v17
	v_mov_b32_e32 v14, v15
                                        ; implicit-def: $sgpr10
                                        ; implicit-def: $sgpr11
                                        ; implicit-def: $sgpr11
	v_mov_b32_e32 v6, s10
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v6
	v_lshlrev_b64 v[14:15], s8, v[14:15]
	v_mov_b32_e32 v6, v15
	v_or_b32_e64 v0, v0, v6
	v_mov_b32_e32 v6, v16
	v_mov_b32_e32 v8, v14
	v_or_b32_e64 v16, v6, v8
                                        ; kill: def $vgpr16 killed $vgpr16 def $vgpr16_vgpr17 killed $exec
	v_mov_b32_e32 v17, v0
	v_mul_hi_u32 v14, v3, v9
                                        ; implicit-def: $sgpr10
	v_mov_b32_e32 v0, s9
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v0
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mov_b32_e32 v6, v14
	v_mov_b32_e32 v8, v15
	v_lshrrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v0, v12
	v_mad_u64_u32 v[14:15], s[10:11], v0, v9, 0
	v_mov_b32_e32 v12, v14
                                        ; implicit-def: $sgpr10
	v_mov_b32_e32 v9, s9
                                        ; kill: def $vgpr12 killed $vgpr12 def $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v9
	v_mov_b32_e32 v9, v13
	v_mov_b32_e32 v14, v15
                                        ; implicit-def: $sgpr10
                                        ; implicit-def: $sgpr11
                                        ; implicit-def: $sgpr11
	v_mov_b32_e32 v16, s10
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v16
	v_lshlrev_b64 v[14:15], s8, v[14:15]
	v_mov_b32_e32 v16, v15
	v_or_b32_e64 v9, v9, v16
                                        ; kill: def $vgpr12 killed $vgpr12 killed $vgpr12_vgpr13 killed $exec
	v_mov_b32_e32 v13, v14
	v_or_b32_e64 v14, v12, v13
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v9
	v_mov_b32_e32 v12, v14
	v_mov_b32_e32 v9, v15
	v_mad_u64_u32 v[14:15], s[10:11], v0, v7, 0
	v_mov_b32_e32 v7, v15
	v_add_co_u32_e32 v6, vcc, v6, v12
	s_nop 1
	v_addc_co_u32_e32 v12, vcc, v8, v9, vcc
	v_mov_b32_e32 v8, s6
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v7, v8, vcc
                                        ; implicit-def: $sgpr10
                                        ; implicit-def: $sgpr11
                                        ; implicit-def: $sgpr11
	v_mov_b32_e32 v7, s10
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v7
	v_lshlrev_b64 v[8:9], s8, v[8:9]
	v_mov_b32_e32 v13, v9
                                        ; kill: def $vgpr14 killed $vgpr14 killed $vgpr14_vgpr15 killed $exec
                                        ; implicit-def: $sgpr10
	v_mov_b32_e32 v7, s9
                                        ; kill: def $vgpr14 killed $vgpr14 def $vgpr14_vgpr15 killed $exec
	v_mov_b32_e32 v15, v7
	v_mov_b32_e32 v7, v15
	v_or_b32_e64 v7, v7, v13
	v_mov_b32_e32 v9, v8
	v_mov_b32_e32 v8, v14
	v_or_b32_e64 v8, v8, v9
                                        ; kill: def $vgpr8 killed $vgpr8 def $vgpr8_vgpr9 killed $exec
	v_mov_b32_e32 v9, v7
                                        ; implicit-def: $sgpr9
                                        ; implicit-def: $sgpr9
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v12
	v_lshrrev_b64 v[6:7], s8, v[6:7]
	v_lshl_add_u64 v[12:13], v[6:7], 0, v[8:9]
	v_mov_b32_e32 v6, v12
	v_mul_lo_u32 v8, v11, v6
	v_lshrrev_b64 v[12:13], s8, v[12:13]
	v_mov_b32_e32 v7, v12
	v_mul_lo_u32 v7, v10, v7
	v_mad_u64_u32 v[12:13], s[8:9], v10, v6, 0
	v_mov_b32_e32 v6, v13
	v_add3_u32 v9, v6, v7, v8
	v_sub_u32_e64 v6, v0, v9
	v_mov_b32_e32 v7, v12
	v_sub_co_u32_e64 v3, s[10:11], v3, v7
	s_nop 1
	v_subb_co_u32_e64 v7, s[8:9], v6, v11, s[10:11]
	v_sub_co_u32_e64 v6, s[12:13], v3, v10
	v_mov_b32_e32 v8, s6
	s_nop 0
	v_subb_co_u32_e64 v8, s[8:9], v7, v8, s[12:13]
	v_cmp_ge_u32_e64 s[8:9], v8, v11
	v_mov_b32_e32 v12, s6
	v_mov_b32_e32 v13, s7
	v_cndmask_b32_e64 v12, v12, v13, s[8:9]
	v_cmp_eq_u32_e64 s[8:9], v8, v11
	v_cmp_ge_u32_e64 s[14:15], v6, v10
	v_mov_b32_e32 v13, s6
	v_mov_b32_e32 v14, s7
	v_cndmask_b32_e64 v13, v13, v14, s[14:15]
	v_cndmask_b32_e64 v12, v12, v13, s[8:9]
	v_cmp_ne_u32_e64 s[8:9], v12, s6
	v_subb_co_u32_e64 v12, s[12:13], v7, v11, s[12:13]
	v_sub_co_u32_e64 v7, s[12:13], v6, v10
	v_mov_b32_e32 v13, s6
	s_nop 0
	v_subb_co_u32_e64 v12, s[12:13], v12, v13, s[12:13]
	v_cndmask_b32_e64 v8, v8, v12, s[8:9]
	v_subb_co_u32_e64 v0, s[10:11], v0, v9, s[10:11]
	v_cmp_ge_u32_e64 s[10:11], v0, v11
	v_mov_b32_e32 v9, s6
	v_mov_b32_e32 v12, s7
	v_cndmask_b32_e64 v9, v9, v12, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], v0, v11
	v_cmp_ge_u32_e64 s[12:13], v3, v10
	v_mov_b32_e32 v10, s6
	v_mov_b32_e32 v11, s7
	v_cndmask_b32_e64 v10, v10, v11, s[12:13]
	v_cndmask_b32_e64 v9, v9, v10, s[10:11]
	v_cmp_ne_u32_e64 s[6:7], v9, s6
	s_nop 1
	v_cndmask_b32_e64 v0, v0, v8, s[6:7]
	v_cndmask_b32_e64 v6, v6, v7, s[8:9]
	v_cndmask_b32_e64 v6, v3, v6, s[6:7]
                                        ; implicit-def: $sgpr6
                                        ; implicit-def: $sgpr6
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v0, v7
	v_xor_b32_e64 v2, v0, v2
	v_mov_b32_e32 v0, v6
	v_xor_b32_e64 v0, v0, v1
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v4
	v_mov_b32_e32 v0, v1
	v_mov_b32_e32 v1, v5
	v_sub_co_u32_e64 v2, s[6:7], v2, v3
	s_nop 1
	v_subb_co_u32_e64 v0, s[6:7], v0, v1, s[6:7]
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v0
	.loc	55 121 17                       ; csrc/cache_kernels_fused.hip:121:17
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_store_dwordx2 v[0:1], v[2:3]
.Ltmp479:
	.loc	55 124 7 is_stmt 1              ; csrc/cache_kernels_fused.hip:124:7
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dwordx2 v[0:1], v[0:1]
	.loc	55 124 16 is_stmt 0             ; csrc/cache_kernels_fused.hip:124:16
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_gt_i64_e64 s[0:1], v[0:1], s[0:1]
	s_mov_b64 s[2:3], exec
	s_and_b64 s[0:1], s[2:3], s[0:1]
	s_xor_b64 s[2:3], s[0:1], s[2:3]
	v_writelane_b32 v43, s2, 34
	s_nop 1
	v_writelane_b32 v43, s3, 35
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB86_9
	s_branch .LBB86_8
.LBB86_7:
.Ltmp480:
	.loc	55 125 5 is_stmt 1              ; csrc/cache_kernels_fused.hip:125:5
	s_branch .LBB86_22
.Ltmp481:
.LBB86_8:
	.loc	56 253 117                      ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:253:117
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_local_id@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_local_id@rel32@hi+12
	v_mov_b32_e32 v0, 0
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s0, v42, 34
	v_readlane_b32 s1, v42, 35
	v_mov_b32_e32 v2, v1
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr2
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, v0
.Ltmp482:
	.loc	55 129 12                       ; csrc/cache_kernels_fused.hip:129:12
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dword v[0:1], v2
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $sgpr2_sgpr3
	.loc	55 129 8 is_stmt 0              ; csrc/cache_kernels_fused.hip:129:8
	v_writelane_b32 v43, s0, 36
	s_nop 1
	v_writelane_b32 v43, s1, 37
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_branch .LBB86_10
.LBB86_9:                               ; %Flow23
	.loc	55 0 8                          ; csrc/cache_kernels_fused.hip:0:8
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s0, v43, 34
	v_readlane_b32 s1, v43, 35
	s_or_saveexec_b64 s[0:1], s[0:1]
	s_and_b64 s[0:1], exec, s[0:1]
	v_writelane_b32 v43, s0, 38
	s_nop 1
	v_writelane_b32 v43, s1, 39
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB86_22
	s_branch .LBB86_7
.LBB86_10:                              ; =>This Inner Loop Header: Depth=1
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s2, v41, 56
	v_readlane_b32 s3, v41, 57
	v_readlane_b32 s4, v42, 34
	v_readlane_b32 s5, v42, 35
	v_readlane_b32 s0, v43, 40
	v_readlane_b32 s1, v43, 41
	v_readlane_b32 s6, v43, 36
	v_readlane_b32 s7, v43, 37
	s_nop 0
	v_writelane_b32 v43, s6, 42
	s_nop 1
	v_writelane_b32 v43, s7, 43
.Ltmp483:
	.loc	55 129 29 is_stmt 1             ; csrc/cache_kernels_fused.hip:129:29
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dword v0, v[0:1]
	.loc	55 129 33 is_stmt 0             ; csrc/cache_kernels_fused.hip:129:33
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v1, v[2:3]
	.loc	55 129 31                       ; csrc/cache_kernels_fused.hip:129:31
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_i32_e64 s[2:3], v0, v1
	s_mov_b64 s[4:5], -1
	s_or_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v43, s0, 44
	s_nop 1
	v_writelane_b32 v43, s1, 45
.Ltmp484:
	.loc	55 129 3                        ; csrc/cache_kernels_fused.hip:129:3
	v_writelane_b32 v43, s0, 46
	s_nop 1
	v_writelane_b32 v43, s1, 47
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v43, s0, 48
	s_nop 1
	v_writelane_b32 v43, s1, 49
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB86_12
; %bb.11:                               ;   in Loop: Header=BB86_10 Depth=1
.Ltmp485:
	.loc	55 130 20 is_stmt 1             ; csrc/cache_kernels_fused.hip:130:20
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s38, v42, 46
	v_readlane_b32 s39, v42, 47
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s10, v41, 3
	v_readlane_b32 s11, v41, 4
	v_readlane_b32 s6, v41, 7
	v_readlane_b32 s7, v41, 8
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s36, v42, 56
	v_readlane_b32 s37, v42, 57
	v_readlane_b32 s26, v42, 44
	v_readlane_b32 s27, v42, 45
	v_readlane_b32 s34, v41, 36
	v_readlane_b32 s35, v41, 37
	v_readlane_b32 s24, v42, 42
	v_readlane_b32 s25, v42, 43
	v_readlane_b32 s8, v42, 40
	v_readlane_b32 s9, v42, 41
	v_readlane_b32 s22, v42, 48
	v_readlane_b32 s23, v42, 49
	v_readlane_b32 s20, v42, 38
	v_readlane_b32 s21, v42, 39
	v_readlane_b32 s16, v42, 50
	v_readlane_b32 s17, v42, 51
	v_readlane_b32 s0, v42, 54
	v_readlane_b32 s1, v42, 55
	v_readlane_b32 s18, v41, 5
	v_readlane_b32 s19, v41, 6
	v_readlane_b32 s28, v42, 36
	v_readlane_b32 s29, v42, 37
	v_readlane_b32 s30, v41, 56
	v_readlane_b32 s31, v41, 57
	v_readlane_b32 s40, v41, 32
	v_readlane_b32 s41, v41, 33
	v_readlane_b32 s42, v41, 48
	v_readlane_b32 s43, v41, 49
	v_readlane_b32 s44, v41, 20
	v_readlane_b32 s45, v41, 21
	v_readlane_b32 s46, v41, 52
	v_readlane_b32 s47, v41, 53
	v_readlane_b32 s2, v42, 34
	v_readlane_b32 s3, v42, 35
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dword v2, v[0:1]
	.loc	55 130 9 is_stmt 0              ; csrc/cache_kernels_fused.hip:130:9
	v_mov_b64_e32 v[0:1], s[28:29]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dword v[0:1], v2
	.loc	55 132 18 is_stmt 1             ; csrc/cache_kernels_fused.hip:132:18
	v_mov_b64_e32 v[0:1], s[46:47]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 132 32 is_stmt 0             ; csrc/cache_kernels_fused.hip:132:32
	v_mov_b64_e32 v[0:1], s[28:29]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	s_mov_b32 s3, 1
	.loc	55 132 30                       ; csrc/cache_kernels_fused.hip:132:30
	v_writelane_b32 v43, s3, 50
	v_lshl_add_u64 v[0:1], v[0:1], s3, v[2:3]
	.loc	55 132 16                       ; csrc/cache_kernels_fused.hip:132:16
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[20:21]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 133 18 is_stmt 1             ; csrc/cache_kernels_fused.hip:133:18
	v_mov_b64_e32 v[0:1], s[46:47]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 133 32 is_stmt 0             ; csrc/cache_kernels_fused.hip:133:32
	v_mov_b64_e32 v[0:1], s[28:29]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 133 30                       ; csrc/cache_kernels_fused.hip:133:30
	v_lshl_add_u64 v[2:3], v[0:1], s3, v[2:3]
	.loc	55 133 43                       ; csrc/cache_kernels_fused.hip:133:43
	v_mov_b64_e32 v[0:1], s[30:31]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 133 41                       ; csrc/cache_kernels_fused.hip:133:41
	v_lshl_add_u64 v[0:1], v[0:1], s3, v[2:3]
	.loc	55 133 16                       ; csrc/cache_kernels_fused.hip:133:16
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[8:9]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 135 27 is_stmt 1             ; csrc/cache_kernels_fused.hip:135:27
	v_mov_b64_e32 v[0:1], s[44:45]
	flat_load_dwordx2 v[0:1], v[0:1]
	.loc	55 135 34 is_stmt 0             ; csrc/cache_kernels_fused.hip:135:34
	v_mov_b64_e32 v[2:3], s[42:43]
	flat_load_dwordx2 v[8:9], v[2:3]
	.loc	55 135 46                       ; csrc/cache_kernels_fused.hip:135:46
	v_mov_b64_e32 v[2:3], s[40:41]
	flat_load_dwordx2 v[2:3], v[2:3]
	s_mov_b32 s2, 32
	.loc	55 135 44                       ; csrc/cache_kernels_fused.hip:135:44
	v_writelane_b32 v43, s2, 51
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshrrev_b64 v[4:5], s2, v[8:9]
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v2
	v_mul_lo_u32 v6, v5, v4
	v_lshrrev_b64 v[2:3], s2, v[2:3]
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v8
	v_mul_lo_u32 v3, v2, v3
	v_mad_u64_u32 v[4:5], s[40:41], v2, v4, 0
	v_mov_b32_e32 v2, v5
	v_add3_u32 v2, v2, v3, v6
                                        ; implicit-def: $sgpr15
                                        ; implicit-def: $sgpr40
                                        ; implicit-def: $sgpr40
	v_mov_b32_e32 v6, s15
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v4
	s_mov_b32 s15, 0
	v_writelane_b32 v43, s15, 52
                                        ; implicit-def: $sgpr40
	v_mov_b32_e32 v4, s15
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v4
	s_mov_b32 s15, 33
	.loc	55 135 32                       ; csrc/cache_kernels_fused.hip:135:32
	v_writelane_b32 v43, s15, 53
	v_lshlrev_b64 v[2:3], s15, v[2:3]
	v_mov_b32_e32 v4, v3
	v_lshlrev_b64 v[6:7], s3, v[6:7]
	v_mov_b32_e32 v5, v7
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[2:3]
	.loc	55 135 11                       ; csrc/cache_kernels_fused.hip:135:11
	v_mov_b64_e32 v[0:1], s[24:25]
	flat_store_dwordx2 v[0:1], v[2:3]
.Ltmp486:
	.loc	55 140 20 is_stmt 1             ; csrc/cache_kernels_fused.hip:140:20
	v_mov_b64_e32 v[0:1], s[28:29]
	flat_load_dword v2, v[0:1]
	.loc	55 140 18 is_stmt 0             ; csrc/cache_kernels_fused.hip:140:18
	v_mov_b64_e32 v[0:1], s[26:27]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_dword v[0:1], v2
	.loc	55 141 20 is_stmt 1             ; csrc/cache_kernels_fused.hip:141:20
	v_mov_b64_e32 v[0:1], s[30:31]
	flat_load_dword v0, v[0:1]
	.loc	55 141 32 is_stmt 0             ; csrc/cache_kernels_fused.hip:141:32
	v_mov_b64_e32 v[2:3], s[28:29]
	flat_load_dword v1, v[2:3]
	.loc	55 141 30                       ; csrc/cache_kernels_fused.hip:141:30
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e64 v2, v0, v1
	.loc	55 141 18                       ; csrc/cache_kernels_fused.hip:141:18
	v_mov_b64_e32 v[0:1], s[38:39]
	flat_store_dword v[0:1], v2
.Ltmp487:
	.loc	55 148 18 is_stmt 1             ; csrc/cache_kernels_fused.hip:148:18
	v_mov_b64_e32 v[0:1], s[24:25]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 148 32 is_stmt 0             ; csrc/cache_kernels_fused.hip:148:32
	v_mov_b64_e32 v[0:1], s[26:27]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 148 18                       ; csrc/cache_kernels_fused.hip:148:18
	v_lshl_add_u64 v[0:1], v[0:1], s3, v[2:3]
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[22:23]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 149 18 is_stmt 1             ; csrc/cache_kernels_fused.hip:149:18
	v_mov_b64_e32 v[0:1], s[24:25]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 149 32 is_stmt 0             ; csrc/cache_kernels_fused.hip:149:32
	v_mov_b64_e32 v[0:1], s[38:39]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 149 18                       ; csrc/cache_kernels_fused.hip:149:18
	v_lshl_add_u64 v[0:1], v[0:1], s3, v[2:3]
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[16:17]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 151 18 is_stmt 1             ; csrc/cache_kernels_fused.hip:151:18
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[22:23]
	flat_store_dwordx2 v[0:1], v[2:3]
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[20:21]
	flat_store_dwordx2 v[0:1], v[2:3] offset:8
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[16:17]
	flat_store_dwordx2 v[0:1], v[2:3] offset:16
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[8:9]
	flat_store_dwordx2 v[0:1], v[2:3] offset:24
	s_mov_b64 s[16:17], 0x80
	s_mov_b32 s8, s18
	s_mov_b32 s3, s19
	s_mov_b32 s15, s16
	s_mov_b32 s9, s17
	s_add_u32 s8, s8, s15
	s_addc_u32 s3, s3, s9
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s3
	v_writelane_b32 v43, s8, 54
	s_nop 1
	v_writelane_b32 v43, s9, 55
	s_lshr_b64 s[2:3], s[0:1], s2
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	s_mov_b32 s3, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv@rel32@lo+4
	s_addc_u32 s1, s1, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv@rel32@hi+12
                                        ; implicit-def: $sgpr15
	v_mov_b32_e32 v0, s3
	v_mov_b32_e32 v1, s2
	s_swappc_b64 s[30:31], s[0:1]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	v_readlane_b32 s22, v42, 50
	v_readlane_b32 s23, v42, 51
	v_readlane_b32 s20, v42, 38
	v_readlane_b32 s21, v42, 39
	v_readlane_b32 s18, v42, 48
	v_readlane_b32 s19, v42, 49
	v_readlane_b32 s16, v42, 40
	v_readlane_b32 s17, v42, 41
	v_readlane_b32 s0, v42, 58
	v_readlane_b32 s1, v42, 59
	v_readlane_b32 s2, v43, 51
	v_readlane_b32 s24, v42, 52
	v_readlane_b32 s25, v42, 53
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s6, v41, 7
	v_readlane_b32 s7, v41, 8
	v_readlane_b32 s8, v43, 54
	v_readlane_b32 s9, v43, 55
	v_readlane_b32 s10, v41, 3
	v_readlane_b32 s11, v41, 4
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s14, v41, 0
	v_mov_b32_e32 v2, v0
	v_mov_b64_e32 v[0:1], s[24:25]
	flat_store_short v[0:1], v2
	.loc	55 158 18                       ; csrc/cache_kernels_fused.hip:158:18
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[22:23]
	flat_store_dwordx2 v[0:1], v[2:3]
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[20:21]
	flat_store_dwordx2 v[0:1], v[2:3] offset:8
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[18:19]
	flat_store_dwordx2 v[0:1], v[2:3] offset:16
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[16:17]
	flat_store_dwordx2 v[0:1], v[2:3] offset:24
	s_lshr_b64 s[2:3], s[0:1], s2
                                        ; kill: def $sgpr2 killed $sgpr2 killed $sgpr2_sgpr3
	s_mov_b32 s3, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv@rel32@lo+4
	s_addc_u32 s1, s1, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv@rel32@hi+12
                                        ; implicit-def: $sgpr15
	v_mov_b32_e32 v0, s3
	v_mov_b32_e32 v1, s2
	s_swappc_b64 s[30:31], s[0:1]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	v_readlane_b32 s40, v42, 42
	v_readlane_b32 s41, v42, 43
	v_readlane_b32 s28, v41, 40
	v_readlane_b32 s29, v41, 41
	v_readlane_b32 s26, v42, 32
	v_readlane_b32 s27, v42, 33
	v_readlane_b32 s24, v41, 42
	v_readlane_b32 s25, v41, 43
	v_readlane_b32 s22, v43, 51
	v_readlane_b32 s21, v43, 52
	v_readlane_b32 s20, v43, 53
	v_readlane_b32 s18, v41, 44
	v_readlane_b32 s19, v41, 45
	v_readlane_b32 s2, v42, 52
	v_readlane_b32 s3, v42, 53
	v_readlane_b32 s0, v42, 62
	v_readlane_b32 s1, v42, 63
	v_readlane_b32 s42, v42, 44
	v_readlane_b32 s43, v42, 45
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s6, v41, 7
	v_readlane_b32 s7, v41, 8
	v_readlane_b32 s8, v43, 54
	v_readlane_b32 s9, v43, 55
	v_readlane_b32 s10, v41, 3
	v_readlane_b32 s11, v41, 4
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s16, v42, 60
	v_readlane_b32 s17, v42, 61
	v_readlane_b32 s15, v43, 50
	v_readlane_b32 s30, v42, 30
	v_readlane_b32 s31, v42, 31
	v_mov_b32_e32 v2, v0
	v_mov_b64_e32 v[0:1], s[36:37]
	flat_store_short v[0:1], v2
	.loc	55 166 5                        ; csrc/cache_kernels_fused.hip:166:5
	v_mov_b64_e32 v[0:1], s[40:41]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 166 19 is_stmt 0             ; csrc/cache_kernels_fused.hip:166:19
	v_mov_b64_e32 v[0:1], s[42:43]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 166 5                        ; csrc/cache_kernels_fused.hip:166:5
	v_lshl_add_u64 v[0:1], v[0:1], s15, v[2:3]
	.loc	55 166 31                       ; csrc/cache_kernels_fused.hip:166:31
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_ushort v2, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 167 5 is_stmt 1              ; csrc/cache_kernels_fused.hip:167:5
	v_mov_b64_e32 v[0:1], s[40:41]
	flat_load_dwordx2 v[2:3], v[0:1]
	.loc	55 167 19 is_stmt 0             ; csrc/cache_kernels_fused.hip:167:19
	v_mov_b64_e32 v[0:1], s[38:39]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 167 5                        ; csrc/cache_kernels_fused.hip:167:5
	v_lshl_add_u64 v[0:1], v[0:1], s15, v[2:3]
	.loc	55 167 31                       ; csrc/cache_kernels_fused.hip:167:31
	v_mov_b64_e32 v[2:3], s[36:37]
	flat_load_ushort v2, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 169 29 is_stmt 1             ; csrc/cache_kernels_fused.hip:169:29
	v_mov_b64_e32 v[0:1], s[34:35]
	flat_load_dwordx2 v[0:1], v[0:1]
	.loc	55 169 40 is_stmt 0             ; csrc/cache_kernels_fused.hip:169:40
	v_mov_b64_e32 v[2:3], s[30:31]
	flat_load_dwordx2 v[8:9], v[2:3]
	.loc	55 169 52                       ; csrc/cache_kernels_fused.hip:169:52
	v_mov_b64_e32 v[2:3], s[28:29]
	flat_load_dword v4, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v5, 31, v4
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v5
	.loc	55 169 50                       ; csrc/cache_kernels_fused.hip:169:50
	v_lshrrev_b64 v[6:7], s22, v[8:9]
	v_mov_b32_e32 v5, v6
	v_mul_lo_u32 v6, v5, v4
	v_lshrrev_b64 v[2:3], s22, v[2:3]
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v8
	v_mul_lo_u32 v3, v2, v3
	v_mad_u64_u32 v[4:5], s[28:29], v2, v4, 0
	v_mov_b32_e32 v2, v5
	v_add3_u32 v2, v2, v3, v6
                                        ; implicit-def: $sgpr23
                                        ; implicit-def: $sgpr28
                                        ; implicit-def: $sgpr28
	v_mov_b32_e32 v6, s23
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v4
                                        ; implicit-def: $sgpr23
	v_mov_b32_e32 v4, s21
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v4
	.loc	55 169 38                       ; csrc/cache_kernels_fused.hip:169:38
	v_lshlrev_b64 v[2:3], s20, v[2:3]
	v_mov_b32_e32 v4, v3
	v_lshlrev_b64 v[6:7], s15, v[6:7]
	v_mov_b32_e32 v5, v7
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	.loc	55 170 29 is_stmt 1             ; csrc/cache_kernels_fused.hip:170:29
	v_mov_b64_e32 v[2:3], s[26:27]
	flat_load_dwordx2 v[8:9], v[2:3]
	.loc	55 170 41 is_stmt 0             ; csrc/cache_kernels_fused.hip:170:41
	v_mov_b64_e32 v[2:3], s[24:25]
	flat_load_dword v4, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v5, 31, v4
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v5
	.loc	55 170 39                       ; csrc/cache_kernels_fused.hip:170:39
	v_lshrrev_b64 v[6:7], s22, v[8:9]
	v_mov_b32_e32 v5, v6
	v_mul_lo_u32 v6, v5, v4
	v_lshrrev_b64 v[2:3], s22, v[2:3]
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v8
	v_mul_lo_u32 v3, v2, v3
	v_mad_u64_u32 v[4:5], s[22:23], v2, v4, 0
	v_mov_b32_e32 v2, v5
	v_add3_u32 v2, v2, v3, v6
                                        ; implicit-def: $sgpr22
                                        ; implicit-def: $sgpr23
                                        ; implicit-def: $sgpr23
	v_mov_b32_e32 v6, s22
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v4
                                        ; implicit-def: $sgpr22
	v_mov_b32_e32 v4, s21
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v4
	.loc	55 169 65 is_stmt 1             ; csrc/cache_kernels_fused.hip:169:65
	v_lshlrev_b64 v[2:3], s20, v[2:3]
	v_mov_b32_e32 v4, v3
	v_lshlrev_b64 v[6:7], s15, v[6:7]
	v_mov_b32_e32 v5, v7
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[2:3]
	.loc	55 170 56                       ; csrc/cache_kernels_fused.hip:170:56
	v_mov_b64_e32 v[0:1], s[18:19]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 170 54 is_stmt 0             ; csrc/cache_kernels_fused.hip:170:54
	v_lshl_add_u64 v[2:3], v[0:1], s15, v[2:3]
	.loc	55 169 14 is_stmt 1             ; csrc/cache_kernels_fused.hip:169:14
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_store_dwordx2 v[0:1], v[2:3]
.Ltmp488:
	.loc	55 179 55                       ; csrc/cache_kernels_fused.hip:179:55
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 179 38 is_stmt 0             ; csrc/cache_kernels_fused.hip:179:38
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_load_ushort v0, v[0:1]
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z16__half_as_ushort6__half@rel32@lo+4
	s_addc_u32 s1, s1, _Z16__half_as_ushort6__half@rel32@hi+12
	v_writelane_b32 v43, s0, 56
	s_nop 1
	v_writelane_b32 v43, s1, 57
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
                                        ; implicit-def: $sgpr15
	s_swappc_b64 s[30:31], s[0:1]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	v_readlane_b32 s18, v42, 44
	v_readlane_b32 s19, v42, 45
	v_readlane_b32 s16, v42, 56
	v_readlane_b32 s17, v42, 57
	v_readlane_b32 s2, v43, 0
	v_readlane_b32 s3, v43, 1
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s6, v41, 7
	v_readlane_b32 s7, v41, 8
	v_readlane_b32 s8, v43, 54
	v_readlane_b32 s9, v43, 55
	v_readlane_b32 s10, v41, 3
	v_readlane_b32 s11, v41, 4
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s0, v43, 56
	v_readlane_b32 s1, v43, 57
	v_readlane_b32 s20, v42, 60
	v_readlane_b32 s21, v42, 61
	v_readlane_b32 s15, v43, 50
	v_mov_b32_e32 v2, v0
	.loc	55 179 11                       ; csrc/cache_kernels_fused.hip:179:11
	v_mov_b64_e32 v[0:1], s[20:21]
	flat_load_dwordx2 v[4:5], v[0:1]
	.loc	55 179 24                       ; csrc/cache_kernels_fused.hip:179:24
	v_mov_b64_e32 v[0:1], s[18:19]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v3, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	.loc	55 179 11                       ; csrc/cache_kernels_fused.hip:179:11
	v_lshl_add_u64 v[0:1], v[0:1], s15, v[4:5]
	.loc	55 179 36                       ; csrc/cache_kernels_fused.hip:179:36
	flat_store_short v[0:1], v2
	.loc	55 180 55 is_stmt 1             ; csrc/cache_kernels_fused.hip:180:55
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 180 38 is_stmt 0             ; csrc/cache_kernels_fused.hip:180:38
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_ushort v0, v[0:1]
                                        ; implicit-def: $sgpr15
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s4, v42, 60
	v_readlane_b32 s5, v42, 61
	v_readlane_b32 s2, v42, 46
	v_readlane_b32 s3, v42, 47
	v_readlane_b32 s0, v43, 50
	v_mov_b32_e32 v2, v0
	.loc	55 180 11                       ; csrc/cache_kernels_fused.hip:180:11
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dwordx2 v[4:5], v[0:1]
	.loc	55 180 24                       ; csrc/cache_kernels_fused.hip:180:24
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v3, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	.loc	55 180 11                       ; csrc/cache_kernels_fused.hip:180:11
	v_lshl_add_u64 v[0:1], v[0:1], s0, v[4:5]
	.loc	55 180 36                       ; csrc/cache_kernels_fused.hip:180:36
	flat_store_short v[0:1], v2
.Ltmp489:
	.loc	55 223 3 is_stmt 1              ; csrc/cache_kernels_fused.hip:223:3
	s_branch .LBB86_13
.Ltmp490:
.LBB86_12:                              ; %Flow22
                                        ;   in Loop: Header=BB86_10 Depth=1
	.loc	55 0 3 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:3
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s0, v43, 48
	v_readlane_b32 s1, v43, 49
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s4, v43, 42
	v_readlane_b32 s5, v43, 43
	v_readlane_b32 s2, v43, 46
	v_readlane_b32 s3, v43, 47
	s_mov_b64 s[0:1], s[2:3]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_writelane_b32 v43, s2, 40
	s_nop 1
	v_writelane_b32 v43, s3, 41
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v43, s2, 36
	s_nop 1
	v_writelane_b32 v43, s3, 37
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v43, s2, 58
	s_nop 1
	v_writelane_b32 v43, s3, 59
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB86_10
	s_branch .LBB86_14
.LBB86_13:                              ;   in Loop: Header=BB86_10 Depth=1
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s0, v41, 5
	v_readlane_b32 s1, v41, 6
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_mov_b64 s[6:7], 0x80
.Ltmp491:
	.loc	56 263 116 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:263:116
	s_mov_b32 s2, s0
	s_mov_b32 s0, s1
	s_mov_b32 s3, s6
	s_mov_b32 s1, s7
	s_add_u32 s8, s2, s3
	s_addc_u32 s0, s0, s1
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_local_size@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_local_size@rel32@hi+12
	v_mov_b32_e32 v0, 0
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s2, v42, 34
	v_readlane_b32 s3, v42, 35
	v_readlane_b32 s0, v43, 44
	v_readlane_b32 s1, v43, 45
	v_mov_b32_e32 v2, v1
                                        ; implicit-def: $sgpr4
                                        ; implicit-def: $sgpr4
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v1, v0
.Ltmp492:
	.loc	55 129 46                       ; csrc/cache_kernels_fused.hip:129:46
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v0, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e64 v2, v0, v1
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_store_dword v[0:1], v2
	s_mov_b64 s[2:3], 0
	s_andn2_b64 s[0:1], s[0:1], exec
	.loc	55 129 3 is_stmt 0              ; csrc/cache_kernels_fused.hip:129:3
	v_writelane_b32 v43, s0, 46
	s_nop 1
	v_writelane_b32 v43, s1, 47
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_branch .LBB86_12
.Ltmp493:
.LBB86_14:
	.loc	55 0 3                          ; csrc/cache_kernels_fused.hip:0:3
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s0, v43, 58
	v_readlane_b32 s1, v43, 59
	s_or_b64 exec, exec, s[0:1]
; %bb.15:
.Ltmp494:
	.loc	56 253 117 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:253:117
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_local_id@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_local_id@rel32@hi+12
	v_mov_b32_e32 v0, 0
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s0, v43, 2
	v_readlane_b32 s1, v43, 3
	v_mov_b32_e32 v2, v1
                                        ; implicit-def: $sgpr2
                                        ; implicit-def: $sgpr2
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v2, v0
.Ltmp495:
	.loc	55 226 12                       ; csrc/cache_kernels_fused.hip:226:12
	v_mov_b64_e32 v[0:1], s[0:1]
	flat_store_dword v[0:1], v2
	s_mov_b64 s[0:1], 0
                                        ; implicit-def: $sgpr2_sgpr3
	.loc	55 226 8 is_stmt 0              ; csrc/cache_kernels_fused.hip:226:8
	v_writelane_b32 v43, s0, 60
	s_nop 1
	v_writelane_b32 v43, s1, 61
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
.LBB86_16:                              ; =>This Inner Loop Header: Depth=1
	.loc	55 0 8                          ; csrc/cache_kernels_fused.hip:0:8
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s2, v42, 44
	v_readlane_b32 s3, v42, 45
	v_readlane_b32 s4, v43, 2
	v_readlane_b32 s5, v43, 3
	v_readlane_b32 s0, v43, 62
	v_readlane_b32 s1, v43, 63
	v_readlane_b32 s6, v43, 60
	v_readlane_b32 s7, v43, 61
                                        ; implicit-def: $vgpr43 : SGPR spill to VGPR lane
	s_nop 0
	v_writelane_b32 v43, s6, 0
	s_nop 1
	v_writelane_b32 v43, s7, 1
.Ltmp496:
	.loc	55 226 29 is_stmt 1             ; csrc/cache_kernels_fused.hip:226:29
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dword v0, v[0:1]
	.loc	55 226 33 is_stmt 0             ; csrc/cache_kernels_fused.hip:226:33
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v1, v[2:3]
	.loc	55 226 31                       ; csrc/cache_kernels_fused.hip:226:31
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cmp_lt_i32_e64 s[2:3], v0, v1
	s_mov_b64 s[4:5], -1
	s_or_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v43, s0, 2
	s_nop 1
	v_writelane_b32 v43, s1, 3
.Ltmp497:
	.loc	55 226 3                        ; csrc/cache_kernels_fused.hip:226:3
	v_writelane_b32 v43, s0, 4
	s_nop 1
	v_writelane_b32 v43, s1, 5
	s_mov_b64 s[0:1], exec
	v_writelane_b32 v43, s0, 6
	s_nop 1
	v_writelane_b32 v43, s1, 7
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a38, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_and_b64 s[0:1], s[0:1], s[2:3]
	s_mov_b64 exec, s[0:1]
	s_cbranch_execz .LBB86_18
; %bb.17:                               ;   in Loop: Header=BB86_16 Depth=1
.Ltmp498:
	.loc	55 227 24 is_stmt 1             ; csrc/cache_kernels_fused.hip:227:24
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v40, a34             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s34, v42, 2
	v_readlane_b32 s35, v42, 3
	v_readlane_b32 s16, v42, 6
	v_readlane_b32 s17, v42, 7
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s10, v41, 3
	v_readlane_b32 s11, v41, 4
	v_readlane_b32 s6, v41, 7
	v_readlane_b32 s7, v41, 8
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s0, v41, 5
	v_readlane_b32 s1, v41, 6
	v_readlane_b32 s2, v42, 8
	v_readlane_b32 s3, v42, 9
	v_readlane_b32 s8, v42, 4
	v_readlane_b32 s9, v42, 5
	v_readlane_b32 s22, v41, 42
	v_readlane_b32 s23, v41, 43
	v_readlane_b32 s24, v40, 32
	v_readlane_b32 s25, v40, 33
	v_readlane_b32 s26, v41, 40
	v_readlane_b32 s27, v41, 41
	v_readlane_b32 s28, v40, 30
	v_readlane_b32 s29, v40, 31
	v_readlane_b32 s30, v41, 36
	v_readlane_b32 s31, v41, 37
	v_readlane_b32 s18, v41, 34
	v_readlane_b32 s19, v41, 35
	v_readlane_b32 s20, v41, 48
	v_readlane_b32 s21, v41, 49
	v_readlane_b32 s36, v41, 22
	v_readlane_b32 s37, v41, 23
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a38             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_accvgpr_read_b32 v31, a32             ;  Reload Reuse
	v_mov_b64_e32 v[0:1], s[36:37]
	flat_load_dwordx2 v[0:1], v[0:1]
	.loc	55 227 31 is_stmt 0             ; csrc/cache_kernels_fused.hip:227:31
	v_mov_b64_e32 v[2:3], s[20:21]
	flat_load_dwordx2 v[8:9], v[2:3]
	.loc	55 227 43                       ; csrc/cache_kernels_fused.hip:227:43
	v_mov_b64_e32 v[2:3], s[18:19]
	flat_load_dwordx2 v[2:3], v[2:3]
	s_mov_b32 s20, 32
	.loc	55 227 41                       ; csrc/cache_kernels_fused.hip:227:41
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_lshrrev_b64 v[4:5], s20, v[8:9]
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v4, v2
	v_mul_lo_u32 v6, v5, v4
	v_lshrrev_b64 v[2:3], s20, v[2:3]
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v8
	v_mul_lo_u32 v3, v2, v3
	v_mad_u64_u32 v[4:5], s[18:19], v2, v4, 0
	v_mov_b32_e32 v2, v5
	v_add3_u32 v2, v2, v3, v6
                                        ; implicit-def: $sgpr15
                                        ; implicit-def: $sgpr18
                                        ; implicit-def: $sgpr18
	v_mov_b32_e32 v6, s15
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v4
	s_mov_b32 s19, 0
                                        ; implicit-def: $sgpr15
	v_mov_b32_e32 v4, s19
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v4
	s_mov_b32 s18, 33
	.loc	55 227 29                       ; csrc/cache_kernels_fused.hip:227:29
	v_lshlrev_b64 v[2:3], s18, v[2:3]
	v_mov_b32_e32 v4, v3
	s_mov_b32 s15, 1
	v_writelane_b32 v43, s15, 8
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a38, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_lshlrev_b64 v[6:7], s15, v[6:7]
	v_mov_b32_e32 v5, v7
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[2:3]
	.loc	55 227 57                       ; csrc/cache_kernels_fused.hip:227:57
	v_mov_b64_e32 v[0:1], s[34:35]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v4, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v4
	.loc	55 227 55                       ; csrc/cache_kernels_fused.hip:227:55
	v_lshl_add_u64 v[0:1], v[0:1], s15, v[2:3]
	.loc	55 227 22                       ; csrc/cache_kernels_fused.hip:227:22
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[8:9]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 230 9 is_stmt 1              ; csrc/cache_kernels_fused.hip:230:9
	v_mov_b64_e32 v[0:1], s[30:31]
	flat_load_dwordx2 v[0:1], v[0:1]
	.loc	55 230 20 is_stmt 0             ; csrc/cache_kernels_fused.hip:230:20
	v_mov_b64_e32 v[2:3], s[28:29]
	flat_load_dwordx2 v[8:9], v[2:3]
	.loc	55 230 32                       ; csrc/cache_kernels_fused.hip:230:32
	v_mov_b64_e32 v[2:3], s[26:27]
	flat_load_dword v4, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v5, 31, v4
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v5
	.loc	55 230 30                       ; csrc/cache_kernels_fused.hip:230:30
	v_lshrrev_b64 v[6:7], s20, v[8:9]
	v_mov_b32_e32 v5, v6
	v_mul_lo_u32 v6, v5, v4
	v_lshrrev_b64 v[2:3], s20, v[2:3]
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v8
	v_mul_lo_u32 v3, v2, v3
	v_mad_u64_u32 v[4:5], s[26:27], v2, v4, 0
	v_mov_b32_e32 v2, v5
	v_add3_u32 v2, v2, v3, v6
                                        ; implicit-def: $sgpr21
                                        ; implicit-def: $sgpr26
                                        ; implicit-def: $sgpr26
	v_mov_b32_e32 v6, s21
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v4
                                        ; implicit-def: $sgpr21
	v_mov_b32_e32 v4, s19
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v4
	.loc	55 230 18                       ; csrc/cache_kernels_fused.hip:230:18
	v_lshlrev_b64 v[2:3], s18, v[2:3]
	v_mov_b32_e32 v4, v3
	v_lshlrev_b64 v[6:7], s15, v[6:7]
	v_mov_b32_e32 v5, v7
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	.loc	55 230 47                       ; csrc/cache_kernels_fused.hip:230:47
	v_mov_b64_e32 v[2:3], s[24:25]
	flat_load_dwordx2 v[8:9], v[2:3]
	.loc	55 230 59                       ; csrc/cache_kernels_fused.hip:230:59
	v_mov_b64_e32 v[2:3], s[22:23]
	flat_load_dword v4, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v5, 31, v4
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v5
	.loc	55 230 57                       ; csrc/cache_kernels_fused.hip:230:57
	v_lshrrev_b64 v[6:7], s20, v[8:9]
	v_mov_b32_e32 v5, v6
	v_mul_lo_u32 v6, v5, v4
	v_lshrrev_b64 v[2:3], s20, v[2:3]
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v2, v8
	v_mul_lo_u32 v3, v2, v3
	v_mad_u64_u32 v[4:5], s[20:21], v2, v4, 0
	v_mov_b32_e32 v2, v5
	v_add3_u32 v2, v2, v3, v6
                                        ; implicit-def: $sgpr20
                                        ; implicit-def: $sgpr21
                                        ; implicit-def: $sgpr21
	v_mov_b32_e32 v6, s20
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_mov_b32_e32 v6, v4
                                        ; implicit-def: $sgpr20
	v_mov_b32_e32 v4, s19
                                        ; kill: def $vgpr6 killed $vgpr6 def $vgpr6_vgpr7 killed $exec
	v_mov_b32_e32 v7, v4
	.loc	55 230 45                       ; csrc/cache_kernels_fused.hip:230:45
	v_lshlrev_b64 v[2:3], s18, v[2:3]
	v_mov_b32_e32 v4, v3
	v_lshlrev_b64 v[6:7], s15, v[6:7]
	v_mov_b32_e32 v5, v7
	v_or_b32_e64 v4, v4, v5
                                        ; kill: def $vgpr2 killed $vgpr2 killed $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v6
	v_or_b32_e64 v2, v2, v3
                                        ; kill: def $vgpr2 killed $vgpr2 def $vgpr2_vgpr3 killed $exec
	v_mov_b32_e32 v3, v4
	v_lshl_add_u64 v[2:3], v[0:1], 0, v[2:3]
	.loc	55 229 14 is_stmt 1             ; csrc/cache_kernels_fused.hip:229:14
	v_mov_b64_e32 v[0:1], s[16:17]
	flat_store_dwordx2 v[0:1], v[2:3]
.Ltmp499:
	.loc	55 237 46                       ; csrc/cache_kernels_fused.hip:237:46
	v_mov_b64_e32 v[0:1], s[8:9]
	flat_load_ushort v2, v[0:1]
	v_mov_b64_e32 v[0:1], s[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	flat_store_short v[0:1], v2
	.loc	55 237 29 is_stmt 0             ; csrc/cache_kernels_fused.hip:237:29
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_ushort v0, v[0:1]
	s_mov_b64 s[8:9], 0x80
	s_mov_b32 s2, s0
	s_mov_b32 s0, s1
	s_mov_b32 s3, s8
	s_mov_b32 s1, s9
	s_add_u32 s8, s2, s3
	s_addc_u32 s0, s0, s1
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _Z16__half_as_ushort6__half@rel32@lo+4
	s_addc_u32 s1, s1, _Z16__half_as_ushort6__half@rel32@hi+12
                                        ; implicit-def: $sgpr15
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s4, v42, 6
	v_readlane_b32 s5, v42, 7
	v_readlane_b32 s2, v42, 2
	v_readlane_b32 s3, v42, 3
	v_readlane_b32 s0, v43, 8
	v_mov_b32_e32 v2, v0
	.loc	55 237 11                       ; csrc/cache_kernels_fused.hip:237:11
	v_mov_b64_e32 v[0:1], s[4:5]
	flat_load_dwordx2 v[4:5], v[0:1]
	.loc	55 237 24                       ; csrc/cache_kernels_fused.hip:237:24
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_load_dword v0, v[0:1]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_ashrrev_i32_e64 v3, 31, v0
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v3
	.loc	55 237 11                       ; csrc/cache_kernels_fused.hip:237:11
	v_lshl_add_u64 v[0:1], v[0:1], s0, v[4:5]
	.loc	55 237 27                       ; csrc/cache_kernels_fused.hip:237:27
	flat_store_short v[0:1], v2
.Ltmp500:
	.loc	55 262 3 is_stmt 1              ; csrc/cache_kernels_fused.hip:262:3
	s_branch .LBB86_19
.Ltmp501:
.LBB86_18:                              ; %Flow
                                        ;   in Loop: Header=BB86_16 Depth=1
	.loc	55 0 3 is_stmt 0                ; csrc/cache_kernels_fused.hip:0:3
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a38             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s0, v43, 6
	v_readlane_b32 s1, v43, 7
	s_or_b64 exec, exec, s[0:1]
	v_readlane_b32 s4, v43, 0
	v_readlane_b32 s5, v43, 1
	v_readlane_b32 s2, v43, 4
	v_readlane_b32 s3, v43, 5
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_mov_b64 s[0:1], s[2:3]
	s_and_b64 s[0:1], exec, s[0:1]
	s_or_b64 s[0:1], s[0:1], s[4:5]
	v_writelane_b32 v42, s2, 62
	s_nop 1
	v_writelane_b32 v42, s3, 63
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v42, s2, 60
	s_nop 1
	v_writelane_b32 v42, s3, 61
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a36, v42            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_mov_b64 s[2:3], s[0:1]
	v_writelane_b32 v43, s2, 9
	s_nop 1
	v_writelane_b32 v43, s3, 10
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a38, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_andn2_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB86_16
	s_branch .LBB86_20
.LBB86_19:                              ;   in Loop: Header=BB86_16 Depth=1
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v41, a33             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s14, v41, 0
	v_readlane_b32 s13, v41, 1
	v_readlane_b32 s12, v41, 2
	v_readlane_b32 s4, v41, 9
	v_readlane_b32 s5, v41, 10
	v_readlane_b32 s0, v41, 5
	v_readlane_b32 s1, v41, 6
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a38             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v42, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_mov_b64 s[6:7], 0x80
.Ltmp502:
	.loc	56 263 116 is_stmt 1            ; /opt/rocm-7.0.0/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_runtime.h:263:116
	s_mov_b32 s2, s0
	s_mov_b32 s0, s1
	s_mov_b32 s3, s6
	s_mov_b32 s1, s7
	s_add_u32 s8, s2, s3
	s_addc_u32 s0, s0, s1
                                        ; kill: def $sgpr8 killed $sgpr8 def $sgpr8_sgpr9
	s_mov_b32 s9, s0
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, __ockl_get_local_size@rel32@lo+4
	s_addc_u32 s1, s1, __ockl_get_local_size@rel32@hi+12
	v_mov_b32_e32 v0, 0
	s_swappc_b64 s[30:31], s[0:1]
	v_readlane_b32 s2, v42, 2
	v_readlane_b32 s3, v42, 3
	v_readlane_b32 s0, v43, 2
	v_readlane_b32 s1, v43, 3
	v_mov_b32_e32 v2, v1
                                        ; implicit-def: $sgpr4
                                        ; implicit-def: $sgpr4
                                        ; kill: def $vgpr0 killed $vgpr0 def $vgpr0_vgpr1 killed $exec
	v_mov_b32_e32 v1, v2
	v_mov_b32_e32 v1, v0
.Ltmp503:
	.loc	55 226 49                       ; csrc/cache_kernels_fused.hip:226:49
	v_mov_b64_e32 v[2:3], s[2:3]
	flat_load_dword v0, v[2:3]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_add_u32_e64 v2, v0, v1
	v_mov_b64_e32 v[0:1], s[2:3]
	flat_store_dword v[0:1], v2
	s_mov_b64 s[2:3], 0
	s_andn2_b64 s[0:1], s[0:1], exec
	.loc	55 226 3 is_stmt 0              ; csrc/cache_kernels_fused.hip:226:3
	v_writelane_b32 v43, s0, 4
	s_nop 1
	v_writelane_b32 v43, s1, 5
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_write_b32 a38, v43            ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	s_branch .LBB86_18
.Ltmp504:
.LBB86_20:                              ; %Flow21
	.loc	55 0 3                          ; csrc/cache_kernels_fused.hip:0:3
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a38             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s0, v43, 9
	v_readlane_b32 s1, v43, 10
	s_or_b64 exec, exec, s[0:1]
; %bb.21:                               ; %Flow21
	s_branch .LBB86_9
.LBB86_22:
	s_or_saveexec_b64 s[96:97], -1
	v_accvgpr_read_b32 v43, a36             ;  Reload Reuse
	s_mov_b64 exec, s[96:97]
	v_readlane_b32 s0, v43, 38
	v_readlane_b32 s1, v43, 39
	s_or_b64 exec, exec, s[0:1]
	.loc	55 263 1 is_stmt 1              ; csrc/cache_kernels_fused.hip:263:1
	s_endpgm
.Ltmp505:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 696
		.amdhsa_kernarg_size 384
		.amdhsa_user_sgpr_count 8
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 1
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 2
		.amdhsa_next_free_vgpr 83
		.amdhsa_next_free_sgpr 98
		.amdhsa_accum_offset 44
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
.Lfunc_end86:
	.size	_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf, .Lfunc_end86-_ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf
	.cfi_endproc
                                        ; -- End function
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.num_vgpr, max(44, .L__ockl_get_group_id.num_vgpr, .L__ockl_get_local_id.num_vgpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.num_vgpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.num_vgpr, .L__ockl_get_local_size.num_vgpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.num_vgpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.num_vgpr, _Z16__half_as_ushort6__half.num_vgpr)
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.num_agpr, max(39, .L__ockl_get_group_id.num_agpr, .L__ockl_get_local_id.num_agpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.num_agpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.num_agpr, .L__ockl_get_local_size.num_agpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.num_agpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.num_agpr, _Z16__half_as_ushort6__half.num_agpr)
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.numbered_sgpr, max(98, .L__ockl_get_group_id.numbered_sgpr, .L__ockl_get_local_id.numbered_sgpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.numbered_sgpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.numbered_sgpr, .L__ockl_get_local_size.numbered_sgpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.numbered_sgpr, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.numbered_sgpr, _Z16__half_as_ushort6__half.numbered_sgpr)
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.private_seg_size, 576+(max(.L__ockl_get_group_id.private_seg_size, .L__ockl_get_local_id.private_seg_size, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.private_seg_size, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.private_seg_size, .L__ockl_get_local_size.private_seg_size, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.private_seg_size, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.private_seg_size, _Z16__half_as_ushort6__half.private_seg_size))
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.uses_vcc, or(1, .L__ockl_get_group_id.uses_vcc, .L__ockl_get_local_id.uses_vcc, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.uses_vcc, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.uses_vcc, .L__ockl_get_local_size.uses_vcc, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.uses_vcc, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.uses_vcc, _Z16__half_as_ushort6__half.uses_vcc)
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.uses_flat_scratch, or(0, .L__ockl_get_group_id.uses_flat_scratch, .L__ockl_get_local_id.uses_flat_scratch, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.uses_flat_scratch, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.uses_flat_scratch, .L__ockl_get_local_size.uses_flat_scratch, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.uses_flat_scratch, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.uses_flat_scratch, _Z16__half_as_ushort6__half.uses_flat_scratch)
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_dyn_sized_stack, or(0, .L__ockl_get_group_id.has_dyn_sized_stack, .L__ockl_get_local_id.has_dyn_sized_stack, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.has_dyn_sized_stack, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.has_dyn_sized_stack, .L__ockl_get_local_size.has_dyn_sized_stack, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.has_dyn_sized_stack, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.has_dyn_sized_stack, _Z16__half_as_ushort6__half.has_dyn_sized_stack)
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_recursion, or(1, .L__ockl_get_group_id.has_recursion, .L__ockl_get_local_id.has_recursion, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.has_recursion, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.has_recursion, .L__ockl_get_local_size.has_recursion, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.has_recursion, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.has_recursion, _Z16__half_as_ushort6__half.has_recursion)
	.set _ZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKf.has_indirect_call, or(0, .L__ockl_get_group_id.has_indirect_call, .L__ockl_get_local_id.has_indirect_call, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE_clEv.has_indirect_call, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE0_clEv.has_indirect_call, .L__ockl_get_local_size.has_indirect_call, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE1_clEv.has_indirect_call, _ZZN4vllm38concat_and_cache_mla_rope_fused_kernelI6__halfLb1EttLNS_18Fp8KVCacheDataTypeE0EEEvPKlPT_S6_PKS5_S8_illlliPT2_S4_iiiiPKfENKUlvE2_clEv.has_indirect_call, _Z16__half_as_ushort6__half.has_indirect_call)
	.section	.AMDGPU.csdata,"",@progbits