# f3a-hip — kernel HIP de atención decode CORRECTO (control de correctitud de la flota)

**Estado: TERMINADO.**

## Resultado en una línea

Hay un kernel HIP de atención decode en gfx1151 que da el resultado **correcto**:
`max_abs ≤ 1.1e-7` contra referencia PyTorch fp32 en **M=1,2,3,4,5** (tolerancia
2e-2, es decir **5 órdenes de magnitud de margen**), con NSEG ∈ {1,5,16,64} y
KV_PAD ∈ {0,64,128} todos verificados; **81 VGPR, 0 spill, 0 scratch**. El precio
de la no-optimización: **445 µs**, 2.24× más lento que el baseline Triton — es un
control de correctitud, no un candidato de rendimiento.

## Alcance

**Solo el primer hito.** Un kernel correcto, verificado, con el ASM mirado. NSEG y
el padding del stride del KV son `#define` parametrizables y **verificados
funcionalmente**, pero deliberadamente **no barridos ni optimizados** — eso es de
otros agentes. No hay comparación de variantes de `P@V`, ni pipelining, ni
persecución de los 96 VGPR.

---

## Cifras

### Correctitud — el entregable

`Hq=32, Hkv=16, D=256, S=2048`, fp16, NSEG=16, KV_PAD=0. Referencia: PyTorch fp32
denso (`torch.softmax` + `bmm`) sobre los mismos bytes fp16 que lee el kernel.

| M | max_abs | max_rel | veredicto (tol 2e-2) |
|---|---|---|---|
| 1 | 6.519e-08 | 3.469e-05 | **PASS** (×3.1e5 de margen) |
| 2 | 2.235e-08 | 1.484e-05 | **PASS** |
| 3 | 2.049e-08 | 1.784e-05 | **PASS** |
| **4 (caso de estudio)** | **2.421e-08** | **2.042e-05** | **PASS** |
| 5 | 2.421e-08 | 1.924e-05 | **PASS** |

El error es ~1e-7 absoluto, es decir **ruido de fp32**, no error de algoritmo.
El kernel acumula en fp32 igual que la referencia; la única diferencia es el orden
de sumación y el reescalado del online softmax.

### Los parámetros expuestos no rompen nada (M=4, S=2048)

| NSEG | KV_PAD (elems fp16) | max_abs | veredicto |
|---|---|---|---|
| **1** (sin split) | 0 | 8.941e-08 | PASS |
| 5 | 0 | 2.794e-08 | PASS |
| **16** (default) | 0 | 2.421e-08 | PASS |
| 64 | 0 | 2.608e-08 | PASS |
| 16 | 64 (`stride_head`=640 B) | 2.421e-08 | PASS |
| 16 | 128 (`stride_head`=768 B) | 2.421e-08 | PASS |

Dos lecturas útiles para quien haga el barrido:

- **NSEG=1 vs NSEG=64 dan el mismo resultado a 1e-7.** La reducción entre
  segmentos (`m_global`/`alpha_seg`) está bien: si estuviera mal, NSEG=1 (que no
  ejercita el reescalado) pasaría y NSEG=64 fallaría. Es la prueba de que el
  `reduce_segments` es correcto, no solo el bucle interno.
- **KV_PAD no toca un solo bit** (2.421e-08 idéntico en pad 0/64/128). Coincide
  con lo que f0 midió en Triton: el padding es una palanca de rendimiento con
  salida bit-idéntica. **Quien barra KV_PAD puede asumir correctitud y medir
  solo tiempo.**

### Controles negativos — la prueba no es vacua

Dos mutantes del kernel, mismo harness, mismos datos:

| mutante | S | max_abs | max_rel | veredicto del test |
|---|---|---|---|---|
| correcto | 2048 | 2.4e-08 | 2.0e-05 | PASS |
| **máscara causal +1 key** | 2048 | 1.249e-03 | **1.065** | PASS ⚠ |
| **máscara causal +1 key** | **48** | **5.217e-02** | **4.0e+01** | **FAIL** |
| correcto | 48 | 1.043e-07 | 3.9e-05 | PASS |
| **GQA: kv head equivocada** | 2048 | 6.187e-02 | 4.2e+01 | **FAIL** |

**Aviso metodológico para la flota, este es el hallazgo del informe:** a S=2048 un
error de *una sola key* en la máscara causal **pasa la tolerancia de 2e-2**
(1.2e-03 de max_abs). Una key de más entre 2045 mueve el softmax ~1/2048 y se
esconde bajo el umbral. El `max_rel` sí lo delata (1.065 contra 2.0e-05), y a
**S=48** el mismo bug falla limpiamente con 5.2e-02.

> **Recomendación: quien valide un kernel de atención en este proyecto debe (a)
> mirar `max_rel`, no solo `max_abs`, y (b) incluir un caso de S pequeño.** Un
> test que solo mire `max_abs` a S=2048 deja pasar off-by-one en el masking.

Con esos dos añadidos, la correctitud de la máscara causal queda verificada de
forma discriminante.

### Tiempo orientativo (una medida, sin barrido)

M=4, S=2048, NSEG=16, KV_PAD=0, 50 iteraciones, 4 copias rotadas del KV (194 MiB
de working set para que MALL no mienta), `hipEvent`:

| | µs | GiB/s | % del roofline (135.9 µs) |
|---|---|---|---|
| **este kernel (sin optimizar)** | **445.07** | 70.2 | **30.5 %** |
| baseline Triton (f0) | 198.87 | 157.1 | 68.3 % |
| Triton con NSEG ajustado (manager) | 185.0 | — | 73.5 % |
| roofline | 135.87 | 230 | 100 % |

**2.24× más lento que Triton.** Es exactamente lo esperado y **no es un problema
a resolver en este hito**: el kernel usa `global_load_d16_b16` (cargas de 16 bits,
una por lane) en lugar de `global_load_b128`, no tiene prefetch, no usa
`__builtin_nontemporal_load`, y mete 7 `s_barrier` por tile. La ruta de LDS
también está sin tocar. Sirve como cota inferior de lo que da un kernel
estructuralmente correcto pero ingenuo: **el 68.3 % de Triton no es fácil de
igualar por accidente.**

---

## Evidencia de ASM

`llvm-objdump`/`-S` sobre el objeto de dispositivo gfx1151 (`out/attn_m4.s`).

### VGPR y spill — la parte que pidió el encargo

```
.name:                        _Z11decode_attnPKDF16_S0_S0_PfS1_S1_if
.vgpr_count:                  81
.vgpr_spill_count:            0
.sgpr_count:                  40
.sgpr_spill_count:            0
.private_segment_fixed_size:  0        <-- 0 bytes de scratch
.group_segment_fixed_size:    19552    <-- 19.1 KiB de LDS
.max_flat_workgroup_size:     256

.name:                        _Z15reduce_segmentsPKfS0_S0_Pf
.vgpr_count:                  36
.vgpr_spill_count:            0
.private_segment_fixed_size:  0
```

**`grep -cE 'scratch_load|scratch_store'` → 0.** Ni una sola instrucción de
scratch en ninguno de los dos kernels. **No hay spill.**

Barrido de M (único "barrido" que hice, y es de compilación, no de GPU):

| M | VGPR | spill | scratch ops | LDS |
|---|---|---|---|---|
| 1 | 33 | 0 | 0 | 17 560 B |
| 2 | 52 | 0 | 0 | 18 224 B |
| 3 | 64 | 0 | 0 | 18 892 B |
| **4** | **81** | **0** | **0** | **19 552 B** |
| 5 | 70 | 0 | 0 | 20 228 B |

### Por qué 81 VGPR y no 256 — y por qué no es mérito

El manager avisó de que con `D_v=256` es esperable VGPR=256 con spill (F4 midió
173 VGPR solo para el acumulador, 256+56 spill en un kernel completo WMMA). **Este
kernel se queda en 81 VGPR sin spill, y la razón es estructural, no una victoria:**

- **`acc` no es `[MAXM][D_v]` por lane, es `[MAXM]`.** El workgroup son
  `HEAD_DIM = 256` hilos y **cada hilo posee una sola dimensión `d` de la
  salida**. El acumulador por lane son 4 floats (M=4), no 4×256. Los 128 VGPR de
  acumulador del §6.1 aquí no existen: viven repartidos entre 256 lanes.
- **El coste se paga en LDS (19.1 KiB) y en ancho de banda desperdiciado**: K y V
  entran por `global_load_d16_b16` — **2 bytes por lane por instrucción** en vez
  de los 16 de `global_load_b128`. De ahí los 70 GiB/s.

Es decir: este diseño cambia presión de registros por eficiencia de memoria, que
es justo el intercambio equivocado para un kernel memory-bound. **Lo digo
explícitamente para que nadie tome los 81 VGPR como evidencia de que el objetivo
de ≤96 VGPR del §0 es alcanzable con un kernel rápido.** F4 y f2-micro ya midieron
que la ruta con `acc` residente por lane no baja de 97 (dot2) o 153–256 (WMMA).

### Mezcla de instrucciones (`decode_attn`, M=4)

```
 142 s_delay_alu          34 global_load_d16_b16     <-- 2 B/lane, el cuello
  89 s_waitcnt            32 ds_load_2addr_b32
  82 v_add_co_u32         25 ds_store_b16
  82 v_add_co_ci_u32_e64  15 ds_load_b128
  52 v_fma_mix_f32        11 global_load_d16_hi_b16
  32 v_add_f32_e32        11 ds_store_b16_d16_hi
  21 v_exp_f32             7 s_barrier
  16 v_max3_f32            7 buffer_gl0_inv
```

Tres observaciones, todas coherentes con lo que ya sabe la flota:

1. **`v_fma_mix_f32`, 52 de ellas, y cero `v_dot2`/`v_dual_dot2acc`.** Es
   exactamente lo que predice R1 de f4-asm: C idiomático (`(float)a * (float)b`)
   baja a `v_fma_mix_f32` (VOP3P, no emparejable en VOPD). Confirmo el hallazgo de
   F4 desde un kernel que además da el resultado correcto. Quien optimice `Q@K` y
   `P@V` tiene que pasar a `__builtin_amdgcn_fdot2` con ≥2 acumuladores.
2. **Cero `global_load_b128`.** Las 45 cargas de KV son de 16 bits. Es el defecto
   dominante de rendimiento de este kernel y es deliberado (simplicidad).
3. **7 `s_barrier` + 7 `buffer_gl0_inv` por iteración de tile.** El precio de
   hacer el softmax por LDS en vez de por registros/DPP.

### Lo que NO se ve en el ASM y es buena señal

- 0 `scratch_*` (sin spill).
- 0 `v_mov` de copia dentro del bucle de WMMA — porque no hay WMMA: este kernel no
  usa matrix cores en absoluto. Es un kernel puramente VALU.

---

## Hipótesis probadas

| # | hipótesis | veredicto | evidencia |
|---|---|---|---|
| 1 | Online softmax con split-KV da el resultado correcto en gfx1151 | **CONFIRMADA** | 1e-7 en M=1..5 contra PyTorch fp32 |
| 2 | La reducción entre segmentos es correcta independientemente de NSEG | **CONFIRMADA** | NSEG=1/5/16/64 coinciden a 1e-7 entre sí y con la referencia |
| 3 | El padding del stride del KV no cambia el resultado | **CONFIRMADA (bit-idéntica)** | KV_PAD 0/64/128 → `max_abs` idéntico a 2.421e-08. Coincide con f0 |
| 4 | Una tolerancia de 2e-2 sobre `max_abs` basta para validar el kernel | **REFUTADA** | un off-by-one en la máscara causal **pasa** a S=2048 (1.2e-3). Hace falta `max_rel` y/o S pequeño |
| 5 | Con `D_v=256` es inevitable VGPR≈256 con spill | **REFUTADA como enunciado general** | 81 VGPR, 0 spill — pero **solo** porque `acc` está repartido entre 256 lanes, a costa de cargas de 2 B/lane. No contradice a F4 para la ruta `acc`-en-registros |
| 6 | C idiomático en el producto punto baja a `v_fma_mix_f32` (R1 de f4-asm) | **CONFIRMADA** | 52 `v_fma_mix_f32`, 0 `v_dot2`, 0 `v_dual_dot2acc` |

---

## Diseño, en cuatro líneas

Dos kernels, workgroup de `HEAD_DIM=256` hilos, un hilo por dimensión de la
salida. `decode_attn`: grid `(NSEG, HQ)`; cada bloque recorre su rodaja del eje S
en tiles de `TILE=32` keys, mantiene `(m, l, acc)` por token de query, y escribe
los parciales. `reduce_segments`: grid `(M, HQ)`; hace
`m_global = max(m_seg)`, `alpha = exp(m_seg − m_global)`,
`out = Σ(acc·alpha) / Σ(l·alpha)`.

Máscara causal: `jb + r <= (S − M) + m`. GQA: `kvh = h / (HQ/HKV)`.
LDS con `LDS_PAD=8` en la dimensión rápida para romper el conflicto de bancos.

Parámetros expuestos (`#define`, sin tunear): `NSEG`, `KV_PAD`, `TILE`, `LDS_PAD`,
`MAXM`, `HEAD_DIM`, `NUM_Q_HEADS`, `NUM_KV_HEADS`.

---

## Qué NO se pudo probar / no se intentó (por alcance)

1. **Nada de rendimiento.** Una sola medida orientativa, sin barrido, sin
   repeticiones múltiples, sin control de dispersión ni de reloj. Los 445 µs son
   un número de referencia, **no una medida con el protocolo de f0/f2**.
2. **No se barrió NSEG ni KV_PAD.** Solo se verificó que varios valores dan el
   resultado correcto. El barrido de tiempo es de otro agente y este kernel
   probablemente no es el vehículo adecuado (es 2.24× más lento que Triton: el
   ruido de su ineficiencia puede tapar el efecto del padding).
3. **No se comparó `P@V` por WMMA ni por dot2.** El kernel usa VALU idiomático
   a propósito.
4. **No se aplicaron R6 (non-temporal) ni R7 (`s_setprio`) de f4-asm**, aunque el
   ASM confirma que harían falta.
5. **No se probó contra el layout paginado real de vLLM.** El harness usa un
   layout denso `[S, HKV, D+KV_PAD]` propio. La correctitud del algoritmo es
   independiente del layout, pero **conectar esto a vLLM exige reescribir el
   indexado para la block table** (f0 documenta el layout real:
   `[blocks, block_size, kv_heads, 2*head_size]`).
6. **Solo fp16 y solo `D=256, HQ=32, HKV=16`.** No se probó el caso Llama
   (`32/8/128`) ni bf16.
7. **No se validó multi-secuencia ni batch > 1.** El harness es de una secuencia.

---

## Cómo reproducir

```bash
source /scratch/rogarcia/vllm/fleet/fleetenv.sh /scratch/rogarcia/f3-hip
bash /scratch/rogarcia/f3-hip/fleet/hip/run.sh 1 2 3 4 5     # verificación completa
S=2048 NSEG=16 KV_PAD=128 bash /scratch/rogarcia/f3-hip/fleet/hip/run.sh 4
ITERS=50 bash /scratch/rogarcia/f3-hip/fleet/hip/run.sh 4    # con medida de tiempo
```

ASM y VGPR:

```bash
build "$HIPCC" -O3 --offload-arch=gfx1151 --cuda-device-only -S -o /tmp/a.s \
    -DMAXM=4 -DNSEG=16 /scratch/rogarcia/f3-hip/fleet/hip/decode_attn.hip
grep -E '\.vgpr_count|\.vgpr_spill_count|\.private_segment_fixed_size' /tmp/a.s
grep -cE 'scratch_load|scratch_store' /tmp/a.s
```

---

## Recomendaciones para la flota

1. **Usad este kernel como oráculo de correctitud, no como base de optimización.**
   Su estructura (un hilo por dimensión de salida, `acc` repartido entre lanes) es
   la que le da los 81 VGPR y también la que le da los 70 GiB/s. Un kernel rápido
   tendrá otra estructura y otra presión de registros.
2. **Arreglad el criterio de validación antes de validar nada.** `max_abs ≤ 2e-2`
   a S=2048 deja pasar un off-by-one en la máscara causal. Añadid `max_rel` y un
   caso a S≈48. `ref_attn.py` ya reporta ambos.
3. **KV_PAD es seguro por construcción**: salida bit-idéntica verificada. Barredlo
   solo por tiempo.
4. **El ASM confirma R1 de f4-asm desde un kernel funcional**: el producto punto
   idiomático da `v_fma_mix_f32` y cero VOPD. Es el primer sitio donde mirar.

---

## Artefactos

Worktree `/scratch/rogarcia/f3-hip` (nada commiteado, nada tocado fuera de
`fleet/hip/`; no se usó `git clean`).

| ruta | qué es |
|---|---|
| `/scratch/rogarcia/f3-hip/fleet/hip/decode_attn.hip` | **el kernel** + driver de host |
| `/scratch/rogarcia/f3-hip/fleet/hip/ref_attn.py` | referencia PyTorch fp32 (`gen` / `check`, reporta `max_abs` y `max_rel`) |
| `/scratch/rogarcia/f3-hip/fleet/hip/run.sh` | build + verificación M=1..5, con `build`/`measure` de fleetenv |
| `/scratch/rogarcia/f3-hip/fleet/hip/out/attn_m4.s` | ASM gfx1151 del caso de estudio |
| `/scratch/rogarcia/f3-hip/fleet/hip/out/data_m*/` | tensores de entrada + referencia por M |
