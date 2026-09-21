# f1-triton — tuning the Triton 3D decode kernel on gfx1151

## Resultado en una línea

Barridos de NSEG, `num_stages`, `TILE_SIZE`, orden del grid, `num_warps` y
`waves_per_eu` sobre el kernel Triton 3D: **NSEG=5 CONFIRMADA pero solo en el
caso de estudio (−1.7 % sola, −7.1 % combinada con `num_stages=1`)**;
**la profundidad de pipeline va al revés de lo que dice §7.2 — `num_stages=1`
gana y `num_stages≥3` cuesta 55 %**; **TILE=32/64 es 1.5–2.3× más lento**, y
**el swap de ejes del grid pierde 11 %**. Neto aplicado a las tablas de tuning:
**−7.1 % en el caso de estudio y −3.4…−6.8 % en las otras cuatro formas**, con
0 fallos de correctitud en 45 celdas.

---

## Aviso de metodología: dos artefactos encontrados antes de medir

**1. El editable-install apunta a un árbol obsoleto.** `.venv` resuelve `vllm`
vía un meta-path finder a `/scratch/rogarcia/vllm-build/vllm`, que es una
**copia de hace dos semanas sin la rama 3D**. Medir desde el árbol compartido
no ejecutaba el kernel 3D en absoluto: caía al 2D por el gate viejo
`max_seqlen_q > 1`, y daba 281 µs. Todas las cifras de este informe se tomaron
desde el worktree aislado `/scratch/rogarcia/f1-triton` con `PYTHONPATH`, tras
verificar el origen de cada import. **La única medida anterior al aislamiento
(281 µs) era el path 2D y está descartada.**

```
$ bash fleet/f1env.sh python -c "import vllm...triton_unified_attention as m; print(m.__file__)"
/scratch/rogarcia/f1-triton/vllm/v1/attention/ops/triton_unified_attention.py
has_3d_tuning: True   has_f1_hooks: True
```

**2. El benchmark mide bf16, no fp16.** `_create_vllm_config` fija
`ModelConfig(dtype="auto")` y el modelo mock es SmolLM2, cuyo dtype nativo es
**bfloat16**; `BenchmarkConfig.dtype = torch.float16` se ignora para el KV cache.
El ASM lo confirma: 20 × `v_wmma_f32_16x16x16_**bf16**`. Medido el impacto
(`fleet/f1_dtype.py`), es pequeño porque el kernel es memory-bound:

| config | bf16 | fp16 | Δ |
|---|---|---|---|
| baseline | 198.79 µs | 195.74 µs | −1.5 % |
| NSEG=5,ST=1,WPEU=2 | 184.85 µs | 184.58 µs | −0.1 % |

§2.3 del doc de diseño predice «bf16 30–40 % más lento». **REFUTADO en este
régimen**: 18–90× por debajo del ridge point, el path bf16 no se nota.

---

## Cifras

Caso de estudio `Hq=32, Hkv=16, M=4, D=256, S=2048`, bf16, bloque 16,
`--min-working-set-mb 96`, 7 repeticiones, mediana. Roofline del protocolo
230 GiB/s → KV de una capa 32.0 MiB → **135.9 µs**.

| variante | tiempo (µs) | vs baseline | % roofline |
|---|---|---|---|
| baseline Triton 3D (NSEG=16, stages=auto→2) | 199.08 | — | 68.2 % |
| `num_stages=1` | 191.05 | −4.0 % | 71.1 % |
| `num_stages=1, waves_per_eu=2` | 190.47 | −4.3 % | 71.4 % |
| NSEG=10, stages=1, wpeu=2 | 188.34 | −5.4 % | 72.2 % |
| **NSEG=5, stages=1, wpeu=2** | **184.79** | **−7.2 %** | **73.5 %** |
| **tablas aplicadas (sin env vars)** | **185.01** | **−7.1 %** | **73.5 %** |
| NSEG=5 sola | 195.72 | −1.7 % | 69.4 % |
| SWAP3D (ejes §4.2) | 220.02 | **+10.6 %** | 61.8 % |
| TILE=32 | 303.54 | **+52.5 %** | 44.8 % |
| TILE=64 | 452.88 | **+127.5 %** | 30.0 % |
| `num_stages=3` | 308.62 | **+55.0 %** | 44.0 % |
| `num_stages=4` | 284.95 | +43.1 % | 47.7 % |
| `waves_per_eu=8` | 345.43 | +73.5 % | 39.3 % |
| `num_warps=1` | 407.39 | +104.6 % | 33.4 % |

Dispersión (max−min)/mediana ≤ 0.9 % en todas las celdas del caso de estudio.

### Las cinco formas, con las tablas aplicadas

| forma | `Hkv, D, S, M` | roofline | baseline | aplicado | Δ | % roofline |
|---|---|---|---|---|---|---|
| study | 16, 256, 2048, 4 | 135.9 | 199.08 | **185.01** | −7.1 % | 73.5 % |
| study_m1 | 16, 256, 2048, 1 | 135.9 | 190.73 | **184.15** | −3.4 % | 73.8 % |
| study_s8k | 16, 256, 8192, 4 | 543.5 | 738.12 | **687.81** | −6.8 % | 79.0 % |
| l3 | 8, 128, 4096, 4 | 67.9 | 104.99 | **98.92** | −5.8 % | 68.7 % |
| gqa4 | 4, 128, 4096, 4 | 34.0 | 47.37 | **45.40** | −4.2 % | 74.9 % |

**Ninguna forma pasa del 79 % del roofline.** El trabajo no está terminado en el
sentido del §4 del protocolo: el kernel bate al baseline pero deja 21–31 % sobre
la mesa. La palanca que queda es §7.1 (padding del stride), que no es tocable
desde Triton — el layout del KV cache lo fija vLLM.

---

## Hipótesis probadas

| # | hipótesis (§ del doc de diseño) | veredicto | evidencia |
|---|---|---|---|
| 1 | **NSEG=5 en vez de 16** (§3) | **CONFIRMADA solo para `Hkv=16,D=256`** | −1.7 % sola, −7.1 % con stages=1. Pero en `l3` cuesta **+13.0 %** y en `gqa4` **+19.5 %** |
| 2 | **Más profundidad de pipeline** (`num_stages` 4–6, §7.2) | **REFUTADA** | `num_stages=1` es el óptimo. 2 = +4.2 %, 3 = **+55.0 %**, 4 = +43.1 %. ASM: stages=3 spillea **181 VGPR** y triplica `FETCH_SIZE` (33 → 106 MiB) |
| 3 | **TILE 32/64 con D=256** (§6.4) | **REFUTADA** | TILE=32 = +52.5 %, TILE=64 = +127.5 %. ASM: TILE=32 spillea 105 VGPR y duplica `FETCH_SIZE` |
| 4 | **Orden de ejes `H_q`→NSEG→`H_kv`** (§4.2) | **REFUTADA** | swap = **+10.6 %** (study), +11.6 % (m1), +2.6 % (l3). El orden actual del path 3D es `(q_blocks, H_kv, NSEG)` — el eje rápido ya es `H_kv`, no NSEG |
| 5 | **`waves_per_eu`/`num_warps` coherentes con 2–4 waves/SIMD** (§2.2) | **CONFIRMADA (ya lo eran)** | `num_warps=4` y `waves_per_eu∈{2,4}` son el óptimo medido; `wpeu=8` = +73.5 %, `warps=1` = +104.6 %. El tuning existente ya estaba en la ventana buena |
| — | **bf16 30–40 % más lento** (§2.3) | **REFUTADA en este régimen** | 1.5 % en baseline, 0.1 % en el mejor. Memory-bound domina |

### Detalle de la hipótesis 1: NSEG=5 no generaliza

NSEG barrido solo (stages/wpeu al default), 5 reps, mediana en µs:

| NSEG | study | study_m1 | l3 | gqa4 |
|---|---|---|---|---|
| 3 | 198.94 | 198.50 | 114.98 | 76.61 |
| 4 | 198.89 | 197.59 | 103.62 | 58.01 |
| **5** | **195.72** | 194.17 | 118.13 | 49.87 |
| 8 | 210.16 | 206.84 | 104.87 | 47.21 |
| **10** | 198.79 | 194.56 | **99.22** | **45.40** |
| 16 (default) | 198.98 | 190.84 | 109.00 | 48.77 |
| 20 | 198.73 | **188.40** | 107.57 | 49.72 |
| 32 | 201.99 | 189.12 | 107.81 | 56.98 |

El criterio de §3.2 («WG múltiplo de 20 y de 40») **predice bien cuando
`WG_base = q_blocks × H_kv` ya es grande**: con `Hkv=16` el grid base es 16 y
NSEG=5 lo lleva a 80 = 4×20. Con `Hkv=8` el grid base es 16 (`q_blocks=2`) y
NSEG=10 lo lleva a 160 = 8×20 — que es el ganador real ahí. Con `Hkv=4`,
NSEG=10 da 80 y vuelve a ganar. **La regla útil no es «NSEG=5» sino
«elegir NSEG tal que `q_blocks × H_kv × NSEG` sea múltiplo de 20»**; 5 solo es
la respuesta cuando `q_blocks × H_kv = 16`.

Nótese también la anomalía de NSEG=8 en el caso de estudio: 210.16 µs, **peor
que 16 y que 5**. 16×8 = 128 WG = 6 rondas de 20 con cola de 8 — el peor punto
de cuantización del barrido, exactamente el efecto que §3.2 predice.

### Detalle de la hipótesis 2: el pipeline ya está saturado

§7.2 dice «4–6 `load_b128` en vuelo por SIMD → 94–97 % del BW» y avisa de que
el default del compilador da 0.7 %. **No es lo que ocurre aquí.** El ASM del
baseline ya emite hasta 8 loads entre `s_waitcnt` consecutivos:

```
[baseline] loads-in-flight per waitcnt window histogram: {1: 2, 2: 2, 4: 1, 6: 1, 8: 1}
[s1w2]     loads-in-flight per waitcnt window histogram: {1: 1, 2: 2, 4: 2}
```

Triton ya hace el pipelining a nivel de MLIR; subir `num_stages` no añade
profundidad útil, añade **presión de registros**, y con D=256 el presupuesto ya
está agotado (§6.4 predice 136 VGPR/lane; el real es 252–256).

---

## Evidencia de ASM / profiler

### Registros y spills (metadatos del `.hsaco`, extraídos del AMDGCN)

| config | `.vgpr_count` | `.vgpr_spill_count` | `.sgpr_count` | scratch (B) | LDS (B) | tiempo |
|---|---|---|---|---|---|---|
| **baseline** (stages=2) | **256** | **6** | 58 | 28 | 16384 | 199.1 |
| **NSEG=5,ST=1,WPEU=2** | **252** | **0** | 52 | **0** | **8192** | **184.8** |
| stages=1 | 252 | 0 | 52 | 0 | 8192 | 191.1 |
| stages=3 | 256 | **181** | 58 | 584 | 24576 | 308.6 |
| stages=4 | 256 | **170** | 58 | 488 | 40960 | 285.0 |
| TILE=32 | 256 | **105** | 74 | 328 | 32768 | 303.5 |
| num_warps=2 | 256 | **129** | 66 | 520 | 16384 | 357.1 |
| waves_per_eu=8 | 192 | **82** | 58 | 332 | 16384 | 345.4 |

Este es el resultado central del ASM: **el tiempo sigue al spill count, no a la
ocupación.** El objetivo de ≤ 96 VGPR/lane del §0 es inalcanzable en Triton con
D=256 — el mínimo observado sin spill es 252. `waves_per_eu=8` sí baja los VGPR
a 192, pero paga 82 spills y es un 73 % más lento: confirma el contra-dato de
§2.2 («no optimizar ocupación; optimizar bytes»).

### Estructura del bucle (`fleet/data/asm/s1w2.s`)

El prólogo agrupa 4 `buffer_load_b128` y los espera en cascada, que es el
patrón que §7.2 recomienda:

```
224:	buffer_load_b128 v[4:7],     v1,  s[20:23], 0 offen
225:	buffer_load_b128 v[8:11],    v2,  s[20:23], 0 offen
226:	buffer_load_b128 v[12:15],   v3,  s[20:23], 0 offen
227:	buffer_load_b128 v[16:19],   v16, s[20:23], 0 offen
...
292:	s_waitcnt vmcnt(3)
294:	s_waitcnt vmcnt(2)
296:	s_waitcnt vmcnt(1)
298:	s_waitcnt vmcnt(0)
```

Pero el bucle interno del KV **no** lo hace: emite 2 loads y espera a 0.

```
495:	buffer_load_b128 v[3:6],     v3, s[24:27], 0 offen
496:	buffer_load_b128 v[219:222], v8, s[24:27], 0 offen
497:	s_waitcnt vmcnt(0)
...
524:	buffer_load_b128 v[3:6],     v3, s[24:27], 0 offen
525:	buffer_load_b128 v[219:222], v7, s[24:27], 0 offen
526:	s_waitcnt vmcnt(0)
```

**§5.4 del protocolo (el `global_load` siguiente inmediatamente tras el
`s_waitcnt`): NO se cumple en ninguna config.** Medido en los 7 volcados,
`waitcnt immediately followed by a load: 0` frente a 11–68 sitios donde hay otra
instrucción en medio. §7.2 cifra ese coste en 10 % — **y no es corregible desde
Triton**: es decisión del scheduler de LLVM. Es material para el agente HIP.

### Bancos de los operandos WMMA (§2.3)

```
=== WMMA dtype ===  20 × v_wmma_f32_16x16x16_bf16
=== bancos (vgpr_start % 4) ===
  A_bank=3 B_bank=1  diff: 16
  A_bank=3 B_bank=3  SAME:  4
```

**Contradice la afirmación de §2.3 de que «HIPCC siempre pone los operandos en
banco 0».** Triton/LLVM aquí los reparte: 16 de 20 WMMA tienen A y B en bancos
distintos (32 cyc), solo 4 colisionan (34 cyc). El margen disponible es
`4/20 × 2/32 ≈ 1.2 %` del tiempo de WMMA, que a 18–90× por debajo del ridge es
ruido. **El 6 % gratis que promete §2.3 no existe en este kernel.**

### Contadores (rocprofv3, mediana de 5 dispatches, `Hq32/Hkv16/D256/S2048`)

| contador | baseline | NSEG=5,ST=1,WPEU=2 | stages=3 | TILE=32 |
|---|---|---|---|---|
| `FETCH_SIZE` | 33.01 MiB | **32.07 MiB** | 105.60 MiB | 62.32 MiB |
| `GL2C_HIT_sum` | 183 832 | 29 625 | 435 664 | 352 481 |
| `GL2C_MISS_sum` | 292 310 | 267 857 | 1 054 851 | 609 773 |
| GL2C hit rate | 38.6 % | **10.0 %** | 29.2 % | 36.6 % |
| `SQ_WAIT_CNT_ANY` | 131.5 M | **67.2 M** | 251.8 M | 195.5 M |
| `SQ_WAIT_ANY` | 178.2 M | **94.9 M** | 306.5 M | 243.9 M |
| `SQ_BUSY_CYCLES` | 10.0 M | 6.5 M | 17.9 M | 16.2 M |
| `SQ_WAIT_CNT_ANY / SQ_WAIT_ANY` | 73.8 % | 70.8 % | 82.1 % | 80.1 % |
| `SQ_WAIT_ANY / SQ_BUSY_CYCLES` | 17.7× | 14.6× | 17.2× | 15.0× |

Lectura:

- **El KV de una capa son 32 MiB y `FETCH_SIZE` del mejor config es 32.07 MiB:
  el kernel lee la KV exactamente una vez.** No hay tráfico redundante que
  recortar — el 26 % que falta para el roofline es ancho de banda efectivo, no
  bytes de más. Esto acota lo que Triton puede dar.
- **Las regresiones son puro re-fetch**: stages=3 lee 105.6 MiB (3.3× la KV) y
  TILE=32 lee 62.3 MiB (1.9×). Ambos spillean a scratch, y el scratch va a
  memoria.
- El mejor config baja el hit rate de L2 del 38.6 % al 10.0 % **y aun así es más
  rápido**: con 80 WG en vez de 256, hay menos reuso incidental en L2 pero
  también menos contención. Confirma que L2 no es la palanca aquí.
- `SQ_WAIT_ANY / SQ_BUSY_CYCLES = 14.6–17.7×`: las waves están esperando ~15×
  más ciclos de los que ejecutan. **Memory-bound, sin ambigüedad.**

**Limitación honesta**: gfx1151 **no expone contadores separados de `vmcnt` y
`lgkmcnt`**. `rocprofv3 --list-avail` solo da `SQ_WAIT_CNT_ANY` (agregado de
todos los contadores), `SQ_WAIT_ANY`, `SQ_WAIT_INST_LDS` y `SQ_WAIT_BARRIER`.
El punto 5 del §5 del protocolo pide «el desglose de stalls por vmcnt vs
lgkmcnt»; **eso no es medible en esta board**. Lo más cercano es
`SQ_WAIT_CNT_ANY / SQ_WAIT_ANY = 70.8–82.1 %`, que dice que la espera en
contadores domina, pero no distingue memoria global de LDS.

### Reparto por kernel (`rocprofv3 --kernel-trace`, 20 capas)

| | `kernel_unified_attention` | `reduce_segments` | suma |
|---|---|---|---|
| baseline (NSEG=16) | 188.41 µs | 4.32 µs | 192.73 µs |
| NSEG=5,ST=1,WPEU=2 | **175.91 µs** | **2.24 µs** | **178.15 µs** |

El coste de los parciales que §3.3 estima en 2 % es **2.3 % medido** con
NSEG=16 y **1.3 %** con NSEG=5. La estimación del documento es correcta.

---

## Correctitud

`fleet/f1_correct.py`: 45 celdas (3 formas × 15 configs), contra referencia
PyTorch fp32 sobre el mismo KV paginado, incluyendo NSEG ∈ {3,4,5,8,10,16,20},
TILE ∈ {32,64}, stages ∈ {2,3,4} y SWAP3D.

```
FAILURES: 0
```

Error máximo absoluto **1.8e-4** (tolerancia 2e-2 para fp16/bf16); error
relativo máximo 4.3e-3. Re-verificado tras aplicar las tablas: 0 fallos.

---

## Cambio de código aplicado

Commit `93ae452` en `rogarcia.fleet-f1-triton`. Tres partes:

1. **`reduce_segments` acepta NSEG no potencia de 2.** `tl.arange` exige
   extensión potencia de 2, así que el kernel iteraba sobre `NUM_SEGMENTS_PER_SEQ`
   directamente y **NSEG=5 ni siquiera compilaba**
   (`ValueError: arange's range must be a power of 2`). Ahora recorre la
   extensión padeada y enmascara la cola; los strides del buffer siguen sobre el
   número real de segmentos. **Sin esto, la hipótesis 1 no es ni siquiera
   ejecutable.**

2. **Entradas `(16, 256)`** en `_GFX1151_3D_SEGMENTS` (→ 5) y
   `_GFX1151_3D_DECODE` (→ `(4, 1, 2, None)`).

3. **`(4, 128)` y `(8, 128)` de 8 a 10 segmentos**, medido −4.2 % y −5.8 %.

Además, hooks `VLLM_F1_*` (`NSEG`, `TILE`, `WARPS`, `STAGES`, `WPEU`, `SWAP3D`)
para el barrido y la variante de orden de ejes del grid 3D. **Son de
instrumentación, no shippables tal cual**: si esto va a PR, los hooks se quitan
y se quedan (1)(2)(3).

---

## Qué NO se pudo probar y por qué

- **Desglose de stalls `vmcnt` vs `lgkmcnt`** (§5.5 del protocolo): gfx1151 no
  expone esos contadores por separado. Solo `SQ_WAIT_CNT_ANY` agregado.
- **Padding del stride del KV** (§7.1, la palanca de 2.55× que el doc llama la
  de mayor impacto): el layout del KV cache lo fija
  `AttentionBackend.get_kv_cache_shape`, no el kernel Triton. Queda fuera del
  encargo de «no reescribir en HIP», pero **es lo que probablemente explica el
  26 % que falta para el roofline** — `FETCH_SIZE` ya es exactamente 1× la KV,
  así que lo que sobra es ancho de banda efectivo, y el stride del caso de
  estudio es 1 MiB exacto, potencia de 2. **Recomendación para la flota: este es
  el siguiente experimento, y no es de Triton.**
- **`global_load` inmediatamente tras `s_waitcnt`** (§5.4): verificado que **no
  se cumple** (0 de 11–68 sitios), pero no es corregible desde Triton.
- **Fusionar `reduce_segments` en el kernel principal** (§11.4): requiere
  reescribir el kernel; fuera del encargo. Su coste medido es 1.3–2.3 %, así que
  el retorno máximo es pequeño.
- **Medición a frecuencia conocida**: no se instrumentó el SCLK durante los
  barridos. Todas las medidas son wall-clock (`triton.testing.do_bench_cudagraph`,
  realtime clock, no `SHADER_CYCLES`), con ≥5 reps y warmup de do_bench, así que
  las comparaciones relativas son válidas; las cifras absolutas corresponden a la
  frecuencia sostenida bajo carga, no a boost.
- **`num_warps` en la tabla**: el barrido dice que 4 es óptimo para
  `Hkv=16,D=256` y 2 para `l3`/`gqa4`, que es lo que las tablas ya ponían. No se
  cambió nada ahí.

---

## Ficheros compartidos tocados

`vllm/v1/attention/ops/triton_unified_attention.py` y
`vllm/v1/attention/backends/triton_attn.py`, **solo dentro del worktree
`/scratch/rogarcia/f1-triton`**. El árbol compartido `/scratch/rogarcia/vllm`
quedó en `gfx11` y sin modificaciones mías.

**Aviso a la flota sobre `reduce_segments`**: el cambio de
`NUM_SEGMENTS_PADDED` toca la firma del kernel de reducción. Cualquier agente
que parta de esta rama lo hereda; cualquiera que mida NSEG no-potencia-de-2 sin
él, se encontrará con el `ValueError` de `tl.arange`.

**Aviso a la flota sobre el editable-install**: medir sin `PYTHONPATH` al propio
worktree ejecuta `/scratch/rogarcia/vllm-build/vllm`, que es código de hace dos
semanas. Cualquier medida de esta flota tomada sin aislamiento debería
revisarse.

---

## Artefactos

Todo en `/scratch/rogarcia/f1-triton/`:

| ruta | qué es |
|---|---|
| `fleet/f1env.sh` | wrapper que garantiza el aislamiento de imports + cache de Triton |
| `fleet/f1_sweep.py` | driver de barridos (`--sweep nseg\|stages\|tile\|grid\|warps\|wpeu\|combo`) |
| `fleet/f1_correct.py` | verificación contra referencia PyTorch, 45 celdas |
| `fleet/f1_asm.py` | volcado de AMDGCN + análisis de VGPR/spill/pipeline |
| `fleet/f1_prof_target.py`, `fleet/f1_prof_run.sh` | recogida de contadores con rocprofv3 |
| `fleet/f1_probe.py` | imprime el launch real (grid, constexprs, config) |
| `fleet/f1_dtype.py` | bf16 vs fp16 |
| `fleet/data/*.json` | medidas crudas de cada barrido (todas las reps) |
| `fleet/data/correctness.txt` | tabla de correctitud completa |
| `fleet/data/asm/*.s` | AMDGCN de 7 configs |
| `fleet/data/prof/*.csv` | contadores crudos |
| commit `93ae452` | el cambio de kernel |
