# F4-asm — Codegen y ASM: qué genera realmente HIPCC en gfx1151

## Resultado en una línea

Siete preguntas de codegen resueltas **compile-only** (cero GPU): VOPD solo se
alcanza con `__builtin_amdgcn_fdot2` y ≥2 acumuladores; **sí se pueden forzar
bancos de VGPR distintos** en WMMA con constraints de registro físico
`"{v[lo:hi]}"` (receta probada, 0 copias extra); la tabla de VGPR del §6.1 está
**confirmada en la forma pero equivocada en los números** (acc D_v=256 wave32
cuesta 173 VGPR, no 128); `num_stages≥2` con D_v=256 entero **es imposible** (56
VGPR de spill ya en stages=1); **`global_load_lds` NO existe en gfx1151**; y la
regla del `s_waitcnt` **sí la viola el compilador** pero `s_setprio` la arregla.

Corrección importante al documento de diseño: el §2.3 dice "A y B deben estar en
bancos distintos". El microbench de referencia (`wmma_bank_latency`) dice que
separar **un solo par no hace nada** — hacen falta **A, B y C los tres en bancos
distintos** para bajar de 34 a 32 cyc. Mi receta lo consigue.

## Cifras

No hay cifras de tiempo: todo mi trabajo es análisis estático. La "cifra" de este
informe es presión de registros y conteos de instrucción.

**Acumulador puro `acc[16, D_v]` fp32** (`q3c_acconly.hip`, medido en
`.vgpr_count` del metadato):

| D_v | wave32 | wave64 | doc §6.1 predice (wave32) |
|---|---|---|---|
| 64  | 77  | —   | 32 |
| 128 | 101 | 71  | 64 |
| 192 | 145 | —   | — |
| 256 | **173** | **103** | **128** |
| 512 | —   | 203 | — |

Ajuste lineal wave32: `VGPR ≈ 45 + 0.5 × D_v`. El término `0.5 × D_v` **es**
exactamente los 128 del documento para D_v=256 (16 fragmentos × 8 floats/lane).
El documento cuenta solo `acc`; el coste real añade ~45 VGPR de direcciones,
operandos en vuelo y temporales. **El tope de 96 VGPR/lane del §6.1 es
inalcanzable con D_v=256 entero en wave32.**

**Kernel de atención completo** (`q3b_regpressure.hip`, online softmax + paged KV):

| variante | VGPR | spill | scratch B | veredicto |
|---|---|---|---|---|
| WMMA, D_v=256 entero, wave32, stages=1 | 256 (tope) | **56** | 228 | **inviable** |
| WMMA, D_v/2, stages=1 | 246 | 0 | 0 | justo |
| WMMA, D_v/4, stages=1 | 213 | 0 | 0 | ok |
| WMMA, D=128 entero, stages=1 | 183 | 0 | 0 | ok |
| dot2/VOPD, D_v=256, stages=1 | **97** | 0 | 0 | **holgado** |
| dot2/VOPD, D_v/2, stages=1 | 83 | 0 | 0 | holgado |

## Hipótesis probadas

| # | hipótesis (§ del doc) | veredicto | evidencia |
|---|---|---|---|
| 1 | `__builtin_amdgcn_fdot2` da VOPD (§2.4) | **CONFIRMADA** (con matiz) | 5–12 `v_dual_dot2acc_f32_f16`; requiere ≥2 acumuladores |
| 1b | inline asm NO empareja (§2.4) | **CONFIRMADA** | 10 `v_dot2_f32_f16` sueltos, 0 duales |
| 1c | C idiomático baja a `v_fma_mix_f32` (§2.4) | **CONFIRMADA** | 20 `v_fma_mix_f32`, 0 dot2 |
| 2a | HIPCC pone WMMA siempre en banco 0 (§2.3) | **CONFIRMADA** (con matiz) | siempre **un solo** banco; a veces b1, no siempre b0 |
| 2b | "A y B en bancos distintos" (§2.3) | **REFUTADA** | hace falta A,B,**C** los tres distintos |
| 2c | se pueden forzar bancos desde HIP | **CONFIRMADA** | `"{v[lo:hi]}"`, 0 `v_mov` extra, prefetch intacto |
| 3 | tabla VGPR 128/64/64 (§6.1) | **REFUTADA en magnitud** | 173/101/97 reales; forma lineal correcta |
| 4 | `num_stages≥2` imposible con D=256 (§6.1) | **CONFIRMADA y peor** | stages=1 ya spillea 56 VGPR |
| 5 | `global_load_lds` disponible (§9) | **REFUTADA** | no existe en gfx1151 ni en ningún RDNA |
| 6 | non-temporal → `slc dlc` (§8.2) | **CONFIRMADA** | `global_load_b128 ... slc dlc` |
| 7 | compilador respeta la regla del `s_waitcnt` (§7.2) | **REFUTADA** | mete 8+ ALU en medio; `s_setprio` lo corrige |

---

# RECETARIO

Para cada técnica: el snippet de HIP exacto y el ASM que lo demuestra.

## R1 — VOPD (`v_dual_dot2acc_f32_f16`)

**La receta.** `__builtin_amdgcn_fdot2(a, b, acc, false)` con **≥ 2
acumuladores independientes**. El segundo requisito es tan obligatorio como el
primero y el documento no lo menciona.

```cpp
float a0 = 0.f, a1 = 0.f;
#pragma unroll 4
for (int i = 0; i < N; ++i) {
    a0 = __builtin_amdgcn_fdot2(pp[2*i+0], pv[2*i+0], a0, false);
    a1 = __builtin_amdgcn_fdot2(pp[2*i+1], pv[2*i+1], a1, false);
}
```

ASM (`out/q1_builtin.s:56-68`):

```
	global_load_b128 v[9:12], v[21:22], off
	global_load_b128 v[13:16], v[17:18], off
	global_load_b128 v[17:20], v[17:18], off offset:16
	global_load_b128 v[21:24], v[21:22], off offset:16
	s_waitcnt vmcnt(2)
	v_dual_dot2acc_f32_f16 v1, v13, v9  :: v_dual_dot2acc_f32_f16 v2, v14, v10
	v_dual_dot2acc_f32_f16 v1, v15, v11 :: v_dual_dot2acc_f32_f16 v2, v16, v12
	s_waitcnt vmcnt(0)
	v_dual_dot2acc_f32_f16 v1, v17, v21 :: v_dual_dot2acc_f32_f16 v2, v18, v22
	v_dual_dot2acc_f32_f16 v1, v19, v23 :: v_dual_dot2acc_f32_f16 v2, v20, v24
```

**Con UN solo acumulador no hay VOPD** — el pairer necesita pareja. Se queda en
la forma ACC single-issue (`out/q1_builtin1.s:56-61`):

```
	v_dot2acc_f32_f16 v5, v8, v12
	v_dot2acc_f32_f16 v5, v9, v13
	v_dot2acc_f32_f16 v5, v10, v14
	v_dot2acc_f32_f16 v5, v11, v15
```

Conteo completo (bucle N-iter, `#pragma unroll 4`):

| variante | `v_dot2_f32_f16` | `v_dot2acc` | `v_dual_dot2acc` | `v_fma_mix` | VGPR |
|---|---:|---:|---:|---:|---:|
| C idiomático (`__half2float`+`fmaf`) | 0 | 0 | **0** | 20 | 25 |
| inline asm `v_dot2_f32_f16` | 10 | 0 | **0** | 0 | 17 |
| `__hfma2` (packed VOP3P) | 0 | 0 | **0** | 2 | 25 |
| **`fdot2`, 1 acumulador** | 0 | 5 | **0** | 0 | 16 |
| **`fdot2`, 2 acumuladores** | 0 | 0 | **5** | 0 | 25 |
| **`fdot2`, 4 acumuladores** | 0 | 0 | **6** | 0 | 25 |
| **`fdot2`, 8 acumuladores** | 0 | 0 | **12** | 0 | 45 |

Lo que **no** funciona, verificado:
- inline asm `v_dot2_f32_f16` — sale literal, opaco al pairer (`out/q1_inline_asm.s:59-60`).
- C idiomático — 20 `v_fma_mix_f32` (VOP3P, no emparejable):
  `v_fma_mix_f32 v2, v14, v10, v2 op_sel_hi:[1,1,0]` (`out/q1_idiomatic.s:61`).
- `__hfma2` — también VOP3P, no empareja.

**Para F2/F3:** en `P@V` con dot2, usad `fdot2` y aseguraos de que el bucle
interno expone ≥2 `acc[j]` independientes por iteración. Con `acc[NACC]` y
`#pragma unroll` interno sale solo.

---

## R2 — Bancos de VGPR en WMMA

### El dato correcto

El §2.3 dice "A y B deben estar en bancos distintos". **Eso es insuficiente.**
Según `rdna35-expert/microbench/wmma_bank_latency` (medido en HW):

| layout | cyc/WMMA |
|---|---|
| todo banco 0 (default del compilador) | 34.02 |
| solo A separado | 34.02 |
| solo B separado | 34.02 |
| solo C separado | 34.02 |
| **A, B y C los tres distintos** | **32.02** |

El 6 % solo aparece si los **tres** están en bancos distintos. Un fix A/B
parcial no da nada.

### Qué hace HIPCC (verificado)

Coloca los tres operandos en **el mismo banco**, siempre. Curiosamente no
siempre en el banco 0 — en `q2c_free` eligió v1/v9/v17/v25 (banco 1). La regla
real es "todos iguales", no "todos en 0":

```
D= 64(b0) A=  0(b0) B=  8(b0) C= 64(b0)   ← q2_free
D=  9(b1) A= 37(b1) B= 45(b1) C=  9(b1)   ← q2c_free
```

### La receta que funciona

Constraints de **registro físico** de LLVM: `"{v[lo:hi]}"`. Clang las acepta
para tuplas de VGPR en inline asm de amdgcn.

**Lo que NO funciona** (probado, errores de compilación):
- `register f16x16 av asm("v[0:7]")` → `error: unknown register name 'v[0:7]'`
- `register f16x16 av asm("v0")` → `error: could not allocate input reg for constraint '{v0}'`
- Meter valores vivos de padding para desplazar al allocator → ignorado, sigue en un banco.

**La receta buena** — un slot por fragmento de acumulador, con A compartida
(bank1), B rotando dentro de bank2, C rotando dentro de bank3:

```cpp
// A = v[1:8]  bank 1  (el fragmento P es el mismo para todos los f)
// B_f: v[10:17], v[22:29], v[34:41], v[46:53]   -> todos bank 2
// C_f: v[55:62], v[67:74], v[79:86], v[91:98]   -> todos bank 3
#define W(CR, BR, C, A, B)                                          \
    asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %0"           \
                 : "+{v[" CR "]}"(C) : "{v[1:8]}"(A), "{v[" BR "]}"(B))

W("55:62","10:17", acc0, pf, v0);
W("67:74","22:29", acc1, pf, v1);
W("79:86","34:41", acc2, pf, v2);
W("91:98","46:53", acc3, pf, v3);
```

ASM resultante (`out/q2e.s`, bucle completo, **sin un solo `v_mov` dentro del
bucle**):

```
.LBB0_2:                                ; =>This Inner Loop Header: Depth=1
	s_clause 0x1
	global_load_b128 v[5:8], v[18:19], off offset:16
	global_load_b128 v[1:4], v[18:19], off
	s_clause 0x7
	global_load_b128 v[14:17], v[20:21], off offset:-3056
	global_load_b128 v[10:13], v[20:21], off offset:-3072
	global_load_b128 v[22:25], v[20:21], off offset:-2048
	global_load_b128 v[26:29], v[20:21], off offset:-2032
	global_load_b128 v[34:37], v[20:21], off offset:-1024
	global_load_b128 v[38:41], v[20:21], off offset:-1008
	global_load_b128 v[46:49], v[20:21], off
	global_load_b128 v[50:53], v[20:21], off offset:16
	...
	s_waitcnt vmcnt(6)
	v_wmma_f32_16x16x16_f16 v[55:62], v[1:8], v[10:17], v[55:62]
	s_waitcnt vmcnt(4)
	v_wmma_f32_16x16x16_f16 v[67:74], v[1:8], v[22:29], v[67:74]
	s_waitcnt vmcnt(2)
	v_wmma_f32_16x16x16_f16 v[79:86], v[1:8], v[34:41], v[79:86]
	s_waitcnt vmcnt(0)
	v_wmma_f32_16x16x16_f16 v[91:98], v[1:8], v[46:53], v[91:98]
```

Bancos: A=1, B=2, C=3 en las cuatro. **Los tres distintos.** 10 loads en vuelo,
prefetch intacto, 99 VGPR frente a 77 del baseline libre (+22 por reservar los
slots).

### La trampa: NO pinees un único slot B

Si todos los fragmentos comparten el mismo registro B, el allocator tiene que
copiarlos (`out/q2d.s`, **28 `v_mov` dentro del bucle**):

```
	s_waitcnt vmcnt(4)
	v_dual_mov_b32 v10, v51 :: v_dual_mov_b32 v11, v52
	v_dual_mov_b32 v12, v53 :: v_dual_mov_b32 v13, v54
	v_dual_mov_b32 v14, v55 :: v_dual_mov_b32 v15, v56
	v_dual_mov_b32 v16, v57 :: v_dual_mov_b32 v17, v58
	v_wmma_f32_16x16x16_f16 v[27:34], v[1:8], v[10:17], v[27:34]
```

16 movs extra por iteración para ahorrar 2 cyc/WMMA × 4 WMMA = 8 cyc. **Pérdida
neta.** La regla: **un slot físico por valor vivo simultáneamente**, nunca
reutilizar el mismo slot dentro de una iteración.

Misma trampa con un solo slot A/B en una cadena secuencial (`out/q2_bankABC.s`):
el prefetch cae de 10 loads en vuelo a 4, con `s_waitcnt vmcnt(0)` entre cada
WMMA. Eso cuesta mucho más del 6 %.

### Resumen para F2/F3

| lo que quieres | cómo | coste |
|---|---|---|
| A/B/C en bancos distintos | `"{v[lo:hi]}"` con slots rotatorios | +20 VGPR aprox. |
| mantener prefetch | un slot físico por fragmento vivo | (incluido arriba) |
| **no hacer** | pinear un slot compartido | 16 `v_mov`/iter o prefetch a 1 |

**Opinión honesta:** es un 6 % sobre la parte WMMA de un kernel que está
18–90× por debajo del ridge y es DRAM-bound. Hacedlo al final, si sobra tiempo,
y midiendo. El riesgo de romper el prefetch es mayor que la ganancia.

---

## R3 — Presión de registros real

Medida = `.vgpr_count` del metadato del `.s` (idéntico al del `.hsaco`).

### Acumulador aislado (`q3c_acconly.hip`)

Lo único vivo grande es `acc[16, D_v]` fp32. K/V entran de uno en uno.

| D_v | wave32 VGPR | wave64 VGPR | acc teórico wave32 |
|---|---|---|---|
| 64  | 77  | —   | 32 |
| 128 | 101 | 71  | 64 |
| 192 | 145 | —   | 96 |
| 256 | **173** | **103** | 128 |
| 512 | —   | 203 | — |

La pendiente es exactamente 0.5 VGPR por unidad de D_v en wave32 (= los
128 del documento para D_v=256) y 0.25 en wave64 (= 64). **La forma de la tabla
del §6.1 es correcta; el offset constante de ~45 VGPR no está contabilizado.**

**wave64 divide `acc` exactamente por 2 — confirmado.** wave64 con D_v=256
(103 VGPR) sale más barato que wave32 con D_v=128 (101 VGPR) y hace el mismo
trabajo por wave. Pero **VOPD es wave32-only**, así que wave64 y la ruta dot2
son mutuamente excluyentes.

### Kernel completo (`q3b_regpressure.hip`)

Online softmax + paged KV + Q en LDS, todo lo que lleva un decode real:

| estrategia | VGPR | spill | scratch | ocupación (waves/SIMD) |
|---|---|---|---|---|
| WMMA D_v=256 entero, w32 | **256** | **56** | 228 B | 2 (inviable) |
| WMMA D_v/2, w32 | 246 | 0 | 0 | 2 |
| WMMA D_v/4, w32 | 213 | 0 | 0 | 2 |
| WMMA D=128 entero, w32 | 183 | 0 | 0 | 2 |
| **dot2/VOPD D_v=256, w32** | **97** | 0 | 0 | **5** |
| **dot2/VOPD D_v/2, w32** | **83** | 0 | 0 | **6** |

**Veredicto sobre la tabla del §6.1:** refutada en magnitud, confirmada en
ordenación relativa. Y el mensaje operativo es más fuerte que el del documento:
con D_v=256 **la ruta WMMA no llega al objetivo de ≤96 VGPR ni partiendo D_v en
4**. La ruta dot2/VOPD sí, y con margen. Si el objetivo de ocupación es real,
`P@V` tiene que ir por dot2.

Aviso metodológico: mi primera versión del kernel daba números absurdos porque
las direcciones eran uniformes y el compilador **escalarizó todo** a
`s_load_b256`. Misma trampa que documenta `microbench/sclause_emission`.
Cualquier medida de presión de registros necesita direccionamiento por lane.

---

## R4 — `num_stages ≥ 2` con D=256

Barrido de `NSTAGE` (tiles KV en vuelo) contra spilling. Señal de spill:
`scratch_load` / `scratch_store` en el ASM.

| DVSPLIT | st=1 | st=2 | st=3 | st=4 |
|---|---|---|---|---|
| 1 (D_v=256 entero) | **56 spill** | 128 spill | 136 spill | 144 spill |
| 2 (D_v/2) | 0 | **24 spill** | 48 spill | 147 spill |
| 4 (D_v/4) | 0 | 0 | 0 | **128 spill** |
| dot2, D_v=256 | 0 | 0 | 0 (st=4) | 0 (st=8) |

**Respuesta: no, y el documento se queda corto.** Con D_v=256 entero en WMMA no
es que stages≥2 sea imposible — es que **stages=1 ya spillea 56 VGPR**. Hay que
partir D_v antes de poder hablar de pipelining.

Frontera práctica verificada:

- **WMMA + D_v/4** → hasta **stages=3** sin spill (235 VGPR). Es el máximo de
  la ruta WMMA.
- **dot2/VOPD, D_v entero** → **stages=8 sin spill** (137 VGPR). El §7.2 pide
  D≈6 loads en vuelo por wave; solo esta ruta lo permite cómodamente.

El coste marginal de un stage en dot2 es ~20 VGPR (st1→st2: 97→121), coherente
con los "4 VGPR por nivel" del §7.2 multiplicados por los ~5 valores en vuelo
por stage de este kernel.

---

## R5 — `global_load_lds` en gfx1151

**NO EXISTE.** Tres comprobaciones independientes:

**1. El ensamblador lo rechaza:**
```
$ llvm-mc -arch=amdgcn -mcpu=gfx1151 <<< 'global_load_lds_dword v1, s[0:1]'
error: instruction not supported on this GPU (gfx1151): global_load_lds_dword

$ llvm-mc -arch=amdgcn -mcpu=gfx942 <<< 'global_load_lds_dword v1, s[0:1]'
	global_load_lds_dword v1, s[0:1]          ← sí en CDNA
```

**2. El builtin de HIP lo rechaza, nombrando la feature que falta:**
```
error: '__builtin_amdgcn_global_load_lds' needs target feature vmem-to-lds-load-insts
```
El mismo fuente compila en gfx942 → `global_load_lds_dword v1, s[0:1]`.

**3. No es cosa de gfx1151, es de toda la familia RDNA.** Probado con llvm-mc:
gfx1100 (RDNA3), gfx1151 (RDNA3.5) y **gfx1200 (RDNA4)** dan todos el mismo
error. `buffer_load ... lds` tampoco: `error: invalid operand for instruction`.
`scratch_load_lds_dword` tampoco.

**Corrección al §9 del documento de diseño.** La línea *"`buffer_load ... lds`
(global → LDS directo, sin pasar por VGPR): ataca el recurso escaso con D=256"*
es **falsa para esta board**. Esa palanca no existe. Global→LDS en gfx1151
obliga a pasar por VGPR: `global_load_b128` → `ds_store_b128`.

Implicación: la presión de registros de R3/R4 **no se puede aliviar por esta
vía**. Las únicas palancas reales sobre `acc` son las tres del §6.1 (wave size,
partir D_v, dot2) — y R4 dice que solo la tercera da margen suficiente.

---

## R6 — Loads non-temporal

**La receta.** `__builtin_nontemporal_load(ptr)` → `slc dlc`. Verificado.

```cpp
typedef unsigned u4 __attribute__((ext_vector_type(4)));
u4 a = __builtin_nontemporal_load(&p[i]);
```

ASM (`out/q6_nt.s`):
```
	s_clause 0x1
	global_load_b128 v[0:3], v[4:5], off slc dlc
	global_load_b128 v[4:7], v[4:5], off offset:128 slc dlc
```
vs. plano (`out/q6_plain.s`):
```
	global_load_b128 v[0:3], v[4:5], off
```

**El flag es por instrucción, no por clause.** Kernel con ambos tipos
mezclados (`out/q6_mixed.s`) — un solo `s_clause 0x3` con los 4 loads, flags
distintos y sin coalescer:
```
	s_clause 0x3
	global_load_b128 v[0:3], v[12:13], off slc dlc          ← NT
	global_load_b128 v[4:7], v[12:13], off offset:64        ← plano
	global_load_b128 v[8:11], v[12:13], off offset:128 slc dlc  ← NT
	global_load_b128 v[12:15], v[12:13], off offset:192     ← plano
```

**Trampa de tipos.** `__builtin_nontemporal_load` **rechaza el `uint4` de HIP**:
```
error: address argument to nontemporal builtin must be a pointer to integer,
float, pointer, or a vector of such types ('const HIP_vector_type<unsigned int,4>*' invalid)
```
Hay que usar un `ext_vector_type` nativo:
```cpp
typedef unsigned u4 __attribute__((ext_vector_type(4)));   // ✓
// typedef uint4 u4;                                        // ✗ rechazado
```

**Stores non-temporal** llevan además GLC (`out/q6_ntstore.s`):
```
	global_store_b128 v[4:5], v[0:3], off glc slc dlc
```

**Si quieres otra combinación de bits** (p.ej. GLC en un load para saltarse L0),
solo llegas por inline asm — y funciona (`out/q6_asmnt.s`):
```cpp
asm volatile("global_load_b128 %0, %1, off glc slc dlc" : "=v"(b) : "v"(pb) : "memory");
```
```
	global_load_b128 v[4:7], v[5:6], off glc slc dlc
```
Coste: pierdes el clausing automático y tienes que poner tú el `s_waitcnt`.

**Para F2/F3:** K/V se leen una vez por paso de decode y no se reutilizan →
`__builtin_nontemporal_load` en todo el streaming de KV, tal como hace wvSplitK
(§8.2). Q sí se reutiliza: Q **sin** el flag, o mejor en LDS.

---

## R7 — La regla del `s_waitcnt`

Regla del §7.2: *"el siguiente `global_load` debe ir INMEDIATAMENTE tras el
`s_waitcnt`; meter ALU de cálculo de dirección en medio cuesta 10 %"*.

**El compilador NO la respeta.** Bucle doble-buffer idiomático
(`out/q7b_plain.s`) — entre el `s_waitcnt vmcnt(0)` y el siguiente burst hay
~20 instrucciones, incluido todo el cálculo de dirección del tile siguiente:

```
.LBB0_2:
	s_waitcnt vmcnt(0)
	v_xor_b32_e32 v10, v1, v2          ← consumo del tile anterior
	...  (≈18 instrucciones de ALU + softmax + direcciones)  ...
	v_add_co_u32 v32, vcc_lo, v18, s10      ← dirección del siguiente
	v_add_co_ci_u32_e64 v33, null, s11, v19, vcc_lo
	s_clause 0x3
	global_load_b128 v[20:23], v[32:33], off slc dlc   ← por fin el load
```

**El fix que funciona: `__builtin_amdgcn_s_setprio`.**

```cpp
__builtin_amdgcn_s_setprio(3);
#pragma unroll
for (int d = 0; d < DEPTH; ++d) b[d] = __builtin_nontemporal_load(&q[d]);
__builtin_amdgcn_s_setprio(0);
```

ASM (`out/q7b_setprio.s`) — el burst sube **por encima** del `s_waitcnt`, que
es aún mejor que "inmediatamente después":

```
.LBB0_2:
	s_load_b64 s[6:7], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s11, s6, 31
	s_lshl_b64 s[10:11], s[10:11], 13
	v_add_co_u32 v32, vcc_lo, v18, s10           ← direcciones ANTES
	v_add_co_ci_u32_e64 v33, null, s11, v19, vcc_lo
	s_setprio 3
	s_clause 0x3
	global_load_b128 v[20:23], v[32:33], off offset:48 slc dlc
	global_load_b128 v[24:27], v[32:33], off offset:32 slc dlc
	global_load_b128 v[28:31], v[32:33], off offset:16 slc dlc
	global_load_b128 v[32:35], v[32:33], off slc dlc
	s_setprio 0
	s_waitcnt vmcnt(4)                            ← espera DESPUÉS de emitir
	v_xor_b32_e32 v13, v13, v14                   ← y ahora el cómputo
```

El `s_waitcnt vmcnt(4)` deja los 4 loads nuevos en vuelo mientras consume los 4
viejos: pipeline perfecto, VGPR idéntico (40 en ambos).

**Comparativa de las tres herramientas** (mismo fuente, DEPTH=4):

| herramienta | ¿burst antes del waitcnt? | clause | notas |
|---|---|---|---|
| nada | **no**, ~20 instr. después | `s_clause 0x3` | el default |
| `s_setprio(3/0)` | **sí** | `s_clause 0x3` | **la receta** |
| `sched_group_barrier(0x020,N,0)` | sí | `s_clause 0x3` | equivalente, más frágil |
| `sched_group_barrier` 1:8 intercalado | sí pero reordena offsets | `s_clause 0x3` | rompe el orden de direcciones |
| `sched_barrier(0)` | sí | rompe el clause | **evitar** |

`s_setprio` es la mejor: es la única que además le dice al HW que priorice la
wave durante el burst, y no interfiere con el `s_clause`. `sched_group_barrier`
da el mismo orden pero hay que acertar con las máscaras (`0x020` = VMEM read,
`0x002` = VALU) y el conteo; si te pasas, reordena los offsets.

**Para F2/F3:** envolved el burst de prefetch de K/V en `s_setprio(3)` /
`s_setprio(0)`. Es una línea, cuesta 0 VGPR, y sin ello el compilador pone el
softmax entero entre el waitcnt y el load siguiente.

---

## Extras verificados de paso

**`s_delay_alu`, no `s_nop`.** HIPCC nunca emite `s_nop` entre WMMAs; usa
hints `s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)`.
Con ≥2 acumuladores independientes los omite (confía en la distancia natural).
Coincide con `microbench/wmma_vgpr_bank`.

**wave64 WMMA existe y funciona.** `__builtin_amdgcn_wmma_f32_16x16x16_f16_w64`
con `-mwavefrontsize64` → `v_wmma_f32_16x16x16_f16 v[17:20], v[1:8], v[9:16],
v[17:20]`. Fragmento de acumulador `f32x4` (4 floats/lane) en vez de `f32x8`.

**El ensamblador acepta cualquier VGPR de arranque** en WMMA — no hay
requisito de alineación a 4 ni a 8. Verificado con llvm-mc:
`v_wmma_f32_16x16x16_f16 v[25:32], v[2:9], v[11:18], v[25:32]` ensambla
limpio (encoding `[0x19,0x40,0x40,0xcc,...]`). Esto es lo que hace posible R2.

**El `s_clause` se parte cada ±2 KB** (offset inmediato de 12 bits con signo),
no en el límite ISA de 63. Confirmado en `out/q2e.s`: offsets de -3072 a +16
sobre dos bases, clauses de 2 y 8.

---

## Qué NO se pudo probar y por qué

- **Nada de esto está medido en GPU.** Todos los veredictos son sobre lo que
  *emite el compilador*, no sobre tiempo. El 6 % de los bancos y el 10 % del
  `s_waitcnt` vienen de los microbenches de `rdna35-expert`, no de mí. Quien
  aplique R2 o R7 debe medir el delta real.
- **No verifiqué corrección numérica** de ningún kernel: son esqueletos de
  codegen, no implementaciones. No calculan atención correcta.
- **`s_setprio` puede tener efectos secundarios de scheduling entre waves** que
  no se ven en ASM. Con varias waves por SIMD podría ser contraproducente.
- **La ruta bf16 no la toqué.** El §2.3 dice que no hay `v_pk_fma_bf16` en
  RDNA3.5; no lo comprobé.
- **La interacción de R2 con NSTAGE>1** no está barrida: reservar slots físicos
  cuando hay 3 tiles en vuelo podría chocar con el allocator.

## Incidencia operativa

**Otro agente de la flota borró `/scratch/rogarcia/vllm/fleet/` a mitad de mi
trabajo** (probablemente un `git clean` en el repo; `fleet/` no está trackeado).
Perdí y tuve que recrear los fuentes de Q1–Q4. He movido el trabajo a
`/scratch/rogarcia/f4-asm-work/asm/` (fuera del repo) y copio a `fleet/`.
**Recomendación para todos: no hagáis `git clean` en `/scratch/rogarcia/vllm`.**

## Artefactos

| ruta | contenido |
|---|---|
| `/scratch/rogarcia/f4-asm-work/asm/` | **fuente canónico** (fuera del repo, a salvo) |
| `/scratch/rogarcia/vllm/fleet/asm/` | copia en el repo |
| `.../asm/sweep.sh` | harness: compila y extrae VGPR/spill/conteos |
| `.../asm/q1_vopd.hip` | 6 variantes VOPD |
| `.../asm/q2_wmma_bank.hip` | bancos WMMA: libre, asm genérico, AB, ABC, rotatorio |
| `.../asm/q2c_bankpractical.hip` | pinning solo A/B (insuficiente — demuestra por qué) |
| `.../asm/q2d_accbank.hip` | slot B compartido (**antipatrón**: 28 `v_mov`/iter) |
| `.../asm/q2e_rotslots.hip` | **la receta buena**: slots rotatorios, 0 copias |
| `.../asm/q3b_regpressure.hip` | kernel de atención completo, paramétrico |
| `.../asm/q3c_acconly.hip` | acumulador aislado, wave32/wave64 |
| `.../asm/q6_nontemporal.hip` | 6 variantes de cache policy |
| `.../asm/q7_waitcnt.hip`, `q7b_pipeline.hip` | scheduling del prefetch |
| `.../asm/out/*.s` | ~40 ficheros de ASM gfx1151 generados |

Reproducir cualquier fila:
```bash
cd /scratch/rogarcia/f4-asm-work/asm
./sweep.sh q3b_regpressure.hip mi_variante -DKERNEL=1 -DDV=256 -DDVSPLIT=2 -DNSTAGE=2
```
(`sweep.sh` ya envuelve hipcc en `quiet-lock build`.)
