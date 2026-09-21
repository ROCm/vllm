# Diseño de kernels de atención — decode y spec decoding en gfx1151

Cómo repartir el trabajo en el régimen `num_seqs=1`, `M ∈ {1..5}`. No cubre cómo
funciona atención; solo decisiones de kernel.

**Caso de estudio:** `Hq=32, Hkv=16, M=4, D=256, S=2048`, fp16.

**Hardware:** Strix Halo (gfx1151) — 40 CU = 20 WGP, 2 SIMD por CU (80 SIMD),
LPDDR5X-8000 a 238.4 GiB/s teóricos.

**Destino:** kernel HIP propio. Las referencias a `triton_unified_attention.py`
son precedente y baseline, no restricción.

**Unidad de despacho:** un WG va a un **WGP** (o CU), nunca a un SIMD. La
cobertura del grid se mide contra **20 WGP / 40 CU**; los 80 SIMD son el suelo de
*waves en vuelo*, un requisito distinto.

Procedencia de los datos: `[M]` medido en gfx1151 (microbenchmarks de
`rdna35-expert` o páginas de AMD MLSE), `[ISA]` de la referencia RDNA3.5, `[D]`
derivado por cálculo, `[?]` sin verificar.

---

## 0. Recomendación (revisada tras medir)

`Hq=32, Hkv=16, M=4, D=256, S=2048` sobre 40 CU / 20 WGP / 80 SIMD.

**Lo que funciona, por impacto medido:**

| # | Decisión | Valor | Impacto | § |
|---|---|---|---|---|
| 1 | **Reparto por lane** | `acc[M][8]`: una fila K/V = un `b128` por lane | **2.67×** | 0.1 |
| 2 | **NSEG (split-KV)** | **1** — no partir el eje KV | 1.10× | 0.1 |
| 3 | **Profundidad de pipeline** | **4** loads en vuelo (no 6) | óptimo del barrido | 0.1 |
| 4 | **K/V** | **caché normal, NO non-temporal** | evita 1.84× de pérdida | 0.1 |
| 5 | **LDS** | **ninguno** — ni para Q | 0 barreras | 0.1 |
| 6 | **Layout del KV** | **`HND` en vez de `NHD`** | **1.07× GRATIS**, acumulable con el kernel | 7.1 |
| 7 | Padding del stride | **descartado** — no transfiere y cuesta bloques | 1.04× por −11 % de VRAM | 7.1 |
| 8 | **`P@V`** | **dot2/VOPD**, no WMMA | 1.57× VALU; **83 vs 153 VGPR** | 5.3 |

**Resultado: 166.96 µs = 81.4 % del roofline**, contra 198.87 µs (68.3 %) del
baseline Triton. Con padding ajustado al harness, 155.55 µs (87.3 %).

**El principio que resume todo:** en este régimen el recurso escaso es la
**localidad de caché**, no el paralelismo del grid. Con GQA=2 se emiten 64 MiB
para 32 MiB únicos, y la caché recupera la mitad — cualquier cosa que rompa ese
reuso (NSEG alto, non-temporal, bloques grandes) cuesta más de lo que aporta.

**Lo que queda:** del 87.3 % al 100 % hay 19.7 µs. `FETCH_SIZE` ya está a 1.03×
del ideal, así que no son bytes: es ancho de banda efectivo y latencia.

---

## 0.1 Resultados medidos por la flota — mandan sobre todo lo demás

Seis agentes han medido las hipótesis de este documento en la board real.
**Kernel HIP + layout HND: 150.20 µs = 90.5 % del roofline, contra 198.65 µs =
68.4 % del baseline. Las dos optimizaciones son ortogonales y se acumulan
(1.32×).**

### Marcador

Matriz 2×2 medida `[F6]`, 3 réplicas A/B alternadas por celda, spread < 0.7 %:

| | NHD | HND | ganancia HND |
|---|---|---|---|
| **Triton** | 198.65 µs · 68.4 % | 185.73 µs · 73.2 % | **1.070×** |
| **HIP paged v5** | 161.40 µs · 84.2 % | **150.20 µs · 90.5 %** | **1.075×** |
| **ganancia kernel** | **1.231×** | **1.237×** | |

**Las dos optimizaciones son ORTOGONALES y se acumulan.** Producto de las
ganancias aisladas 1.316× contra 1.323× medido en combinación: **0.47 % de
desvío**. La ganancia del kernel es la misma en ambos layouts (1.231 vs 1.237) y
la de HND la misma en ambos kernels (1.070 vs 1.075).

**Resultado final: 150.20 µs = 90.5 % del roofline**, 1.32× sobre el baseline.
Quedan 14.3 µs hasta el techo teórico.

Por qué son ortogonales: **HND arregla la dispersión ENTRE kv-heads** — una
propiedad del tensor en memoria (16 workgroups leyendo 16 regiones separadas por
1 KiB), que ningún kernel puede evitar con su patrón de acceso. **El kernel HIP
arregla el INTERIOR del bucle** (NSEG=1, KPW=4, `b128`, 0 barreras, 0 LDS). No
tocan el mismo nivel.

**Corrección de cifra para el PR:** el 1.09× de HND **no se sostiene: es 1.07×**.
La medida original se tomó sin `--min-working-set-mb 96`, mientras el baseline de
198.87 µs sí lo lleva — eran regímenes distintos. La proyección final no cambia
porque el kernel gana más sobre el layout paginado real (161.4 µs) que sobre el
denso (166.96 µs).

**HND solo paga con `H_kv` alto** `[F6]`: 4 → 0.99×, 8 → 1.022×, 16 → 1.073×.
Para modelos tipo Llama (`H_kv=8`) es marginal. Verificado también dentro del
kernel HIP, que es donde estaba la duda.

Robustez: con asignador realista (`shuf=1 poolx=4`) la ganancia se mantiene
(1.069×). NHD y HND dan **salida idéntica**: el layout no toca la aritmética.

### Barrido de contexto: S ∈ {128 … 32768} `[F9]`

28 celdas, `Hq=32, Hkv=16, D=256, M=4`. µs (% del roofline):

| S | Triton NHD | Triton HND | HIP NHD | **HIP HND** |
|---|---|---|---|---|
| 128 | 23.80 (35.7 %) | 22.98 (36.9 %) | **14.10 (60.2 %)** | 14.44 (58.8 %) |
| 1024 | 111.02 (61.2 %) | 103.89 (65.4 %) | 85.21 (79.7 %) | **80.26 (84.6 %)** |
| 2048 | 199.15 (68.2 %) | 185.71 (73.2 %) | 162.61 (83.6 %) | **153.16 (88.7 %)** |
| 4096 | 376.52 (72.2 %) | 350.97 (77.4 %) | 316.60 (85.8 %) | **295.75 (91.9 %)** |
| 8192 | 737.07 (73.7 %) | 689.69 (78.8 %) | 618.70 (87.8 %) | **585.46 (92.8 %)** |
| 16384 | 1441.03 (75.4 %) | 1353.00 (80.3 %) | 1220.56 (89.1 %) | **1157.49 (93.9 %)** |
| 32768 | 2838.81 (76.6 %) | 2644.74 (82.2 %) | 2421.45 (89.8 %) | **2308.08 (94.2 %)** |

**El kernel HIP gana en los siete contextos**, y alcanza **94.2 % del roofline**
a 32k.

**La ventaja es máxima donde menos se esperaba: 1.688× a S=128** frente a 1.172×
a 32k. El mérito principal del kernel es el **overhead fijo bajo** (0 barreras,
0 LDS, NSEG=1), no un bucle interno mejor — Triton se desploma al 35.7 % a S=128
porque el launch y el prólogo dominan.

**El % de roofline no satura en 72 %**: Triton sube monótono hasta 76.6 % a 32k.

**`NSEG=1` gana en todo el rango, y el castigo por subirlo CRECE con S**: a
S=32768, NSEG=2 cuesta 1.22× y NSEG=16 cuesta 1.50×. La hipótesis «más contexto
→ más tiles → más paralelismo → NSEG mayor» queda **refutada**. `reduce_segments`
lee `NSEG × NWAVE` parciales por (m,h) y el paralelismo ya está saturado por
32 heads × 8 waves.

**HND no se diluye en DRAM**: en Triton la ganancia es plana (1.065–1.073×) de
1k a 32k, aunque a 32k el KV (512 MiB) sea **16× el MALL**. En el kernel HIP
decae suavemente (1.070× a 4k → 1.049× a 32k) porque a 94 % del roofline queda
poco margen. **Cifra a citar en contextos largos: ~1.05×, no 1.07×.**

Única celda donde HND pierde: **S=128** (0.976×, reproducido en dos tandas). Con
solo 8 páginas de 16 tokens, el stride de head de HND no se amortiza.

**Sesgo de caché cuantificado**: sin `--min-working-set-mb 96`, S=128 da 17.79 µs
en vez de 23.80 = **+33.8 % optimista**. La rodilla está **exactamente en el MALL
de 32 MiB** (24 MiB de working set → 8.69 µs; 48 MiB → 13.91 µs). Con working
set pequeño el kernel «alcanza» el 103 % del roofline — la señal delatora.

### El kernel generaliza a D ∈ {64, 128, 256} `[F8]`

`D=128` es la config más común (Llama-3, Qwen, Mistral) y el `static_assert(DPL == 8)`
lo excluía. Resuelto **manteniendo `DPL=8` fijo y partiendo la wave**:
`LPR = D/8` lanes cubren una fila y los `SUB = WAVE/LPR` grupos caminan cada uno
su token.

| D | LPR | SUB | |
|---|---|---|---|
| 256 | 32 | 1 | el reparto de v5, exacto |
| 128 | 16 | 2 | |
| 64 | 8 | 4 | validado |

Los streams nunca se encuentran —cada grupo es su propio segmento de salida— así
que la máscara causal no se complica y siguen siendo **0 barreras y 0 LDS**.

| | Triton | HIP v6 | ganancia |
|---|---|---|---|
| **D=128** (`Hq=32,Hkv=8,S=2048`) | 54.95 µs · 61.8 % | **40.70 µs · 83.5 %** | **1.35×** |
| D=256 (no rompe) | — | 168.60 vs 167.35 µs de v5 | — |

128 VGPR, **cero spills**, 12 `global_load_b128`. **`NSEG=2` gana a D=128**, al
revés que a D=256: con `H_kv=8` el grid es la mitad de ancho y a NSEG=1 falta
paralelismo.

Con `H_kv=8` ni HND (1.022×) ni el guard de stride (0.99×) aportan — **a D=128 la
ganancia viene toda del kernel**.

### Lo que de verdad movió la aguja

| Optimización | Ganancia | Nota |
|---|---|---|
| **Rediseño del reparto por lane** | **2.67×** | `acc` de `[M]` a `[M][8]`: una fila de K/V = un `global_load_b128` por lane |
| `NSEG=1` | 1.10× | **al revés de lo que predecía este documento** |
| `KV_PAD=64` | 1.07× | depende del layout; no transferible |

Efecto secundario del rediseño: **0 `s_barrier` y 0 LDS en todo el kernel** (antes
7 barreras por tile). Cada wave es su propio flujo de softmax y escribe su
segmento a global, que `reduce_segments` ya sabía fusionar. 129 VGPR, 0 spill.

### Hipótesis del documento: veredicto final

| # | Hipótesis (§) | Veredicto |
|---|---|---|
| 1 | Stride del KV: 2.55× (§7.1) | **REFUTADA**: el layout actual ya es óptimo. El riesgo real es `stride_head % 2048 == 0` (hasta 3.76×), inalcanzable hoy |
| 2 | NSEG=5 / split-KV alto (§3) | **REFUTADA en HIP**: `NSEG=1` gana. Curva monótona: 165.6 / 167.0 / 260.7 / 324.3 / **700.1** µs para 1/2/4/8/16 |
| 3 | Profundidad 4–6 (§7.2) | **CONFIRMADA en 4**: KPW 2/4/6/8 → 176.0 / **167.2** / 201.5 / 174.6 µs. Cuesta **16 VGPR/nivel**, no 4 |
| 4 | K/V non-temporal (§8.2) | **REFUTADA, es 1.84× PEOR** (319 vs 174 µs) |
| 5 | `s_setprio` (§9) | **NEUTRO** (0.1 %) con 12 `b128` ya en vuelo |
| 6 | Q a LDS (§0) | **INNECESARIO**: el kernel ganador usa **0 LDS** |
| 7 | TILE/bloques grandes (§6.4) | **REFUTADA**: BLOCK 512 → 188 µs, 1024 → 267 µs |
| 8 | `global_load_lds` (§9) | **NO EXISTE** en gfx1151 (CDNA-only) |
| 9 | Bancos WMMA "A y B" (§2.3) | **INSUFICIENTE**: hacen falta A, B **y** C |
| 10 | bf16 30–40 % peor (§2.3) | **REFUTADA**: 1.5 % en memory-bound |

### El hilo que une los tres resultados contraintuitivos

`NSEG=1` gana, el non-temporal **daña** 1.84×, y GQA-fusion no ayuda. Es el mismo
fenómeno: **con GQA=2 dos bloques de q-heads leen la misma kv head**, así que se
emiten 64 MiB para 32 MiB únicos y **la caché recupera la mitad del tráfico**.

- Subir NSEG multiplica el conjunto de trabajo vivo y destruye ese reuso.
- El non-temporal lo destruye explícitamente (por eso es la peor de las medidas).
- El coste **no está en el epílogo**: `reduce_segments` nunca pasa de 14 µs
  mientras la diferencia es de 160 µs.

**Corolario de diseño: en este régimen el recurso a proteger es la localidad de
caché, no el paralelismo del grid.** Todo el §3 de este documento optimiza lo
contrario.

### Metodología

- **El criterio `max_abs ≤ 2e-2` del protocolo era defectuoso.** Un off-by-one en
  la máscara causal da `max_abs = 1.2e-03` a S=2048 y **pasa**; `max_rel` lo
  delata (1.065 vs ~2e-5) y S=48 lo caza con ambas métricas. Validar siempre con
  `max_rel` + un S pequeño + un control negativo.
- El mecanismo del acantilado de stride es **conflicto de sets en caché**, no
  contención de canal DDR: con **1 sola kv-head viva el efecto desaparece**
  (63.7 vs 63.5 GiB/s) y `FETCH_SIZE` no cambia mientras
  `GL1C_STALL_GL2_GL1` va 177 → 1294 → 119.
- El editable-install apunta a un árbol **655 commits atrasado** sin el path 3D:
  medir sin `PYTHONPATH` benchmarkea el kernel equivocado sin dar error.

### Limitación del kernel ganador

`static_assert(DPL == 8)` fija el diseño a 8 fp16 por lane: **D=128 necesitaría
dos lanes por fila** y otro reparto. Válido para D=256.

---

## 1. El régimen

### 1.1 Los ejes disponibles

| Eje | Tamaño en decode | ¿Reparte trabajo? |
|---|---|---|
| secuencias | 1 | no — es el supuesto |
| queries (M) | 1–5 | apenas — no llena ni un q-block |
| cabezas KV | 8–16 | sí, pero es techo fijo del modelo |
| **KV (S/TILE)** | **128–256 tiles** | **sí — el único con material** |
| head_dim (D) | 128–256 | no — profundidad de reducción, no paralelismo |

La última fila es el error fácil: los `D/16` pasos matriciales son la profundidad
del producto, se consumen **dentro** del mismo workgroup.

**El trabajo sale del eje KV o no sale de ningún sitio.**

### 1.2 Intensidad aritmética

Cada elemento de K/V (2 bytes) participa en ~`2·(M·Hq/Hkv)` FLOPs →
**2–10 FLOP/byte**. El ridge point de gfx1151 es `43 TF / 222 GiB/s ≈ 180
FLOP/byte` `[M]`.

Estamos **18–90× por debajo**. Toda decisión de unidad matricial es de segundo
orden frente a los bytes que cruzan el muro.

---

## 2. Datos de hardware medidos (gfx1151)

Base factual del resto del documento.

### 2.1 Jerarquía de memoria `[M]`

| Nivel | Capacidad | Latencia | BW pico |
|---|---|---|---|
| L0 / GL1 | ~16 KiB / 256 KiB `[?]` | 38–43 ns (~110–125 cyc) | 275 GiB/s single-CU |
| **L2** | **2 MiB** | 100–111 ns | **1767 GiB/s** |
| **MALL** | **32 MiB** | 162–179 ns | 925 GiB/s |
| DRAM | — | 516–561 ns (~1500 cyc) | **232 GiB/s** (96 % del teórico) |

Las capacidades de L0 y GL1 son *family defaults*, no verificadas en gfx1151.

**L0/GL1/L2 se invalidan en CADA lanzamiento de kernel** `[M]` (hit rate 87 % →
2 %). **Solo MALL persiste** entre kernels; se desaloja por LRU set-asociativa.

Consecuencia directa: un split-KV en dos kernels (parciales + reducción) **relee
los parciales desde MALL, no desde L2**. Argumento a favor de fusionar la
reducción en el mismo kernel.

### 2.2 Registros y ocupación `[M]`

| Dato | Valor |
|---|---|
| VGPR por SIMD | **1536** (resto de RDNA3.5: 1024) |
| Bloque de asignación | 24 (wave32) / 12 (wave64) |
| **Tope duro de waves/SIMD** | **16** |
| Máx. VGPR por wave | 256 |
| SGPR | nunca limita la ocupación en RDNA |

```
waves/SIMD = min(16, floor(1536 / (ceil(n/24)·24)))       [D]
```

Por debajo de **96 VGPR/lane manda el tope de 16 waves**, no los registros. Por
encima se empieza a perder ocupación: 120 → 12 waves, 144 → 10, 192 → 8.

**Ocupación objetivo para kernels con unidad matricial: 2–4 waves/SIMD**, no el
máximo `[M]`. 1 bloque/CU = 46–48 % del pico (latency-bound); ≥2 bloques =
95–97 %.

Contra-dato relevante `[M]`: en un GEMM real, bajar el tile para subir de 7 a 8
waves/SIMD **empeoró el wall-time un 7–18 %**, porque duplicó los re-fetch desde
DRAM. **No optimizar ocupación; optimizar bytes.**

### 2.3 Unidad matricial `[M]`

Picos **teóricos** por CU:

| Ruta | FLOPs/cyc/CU |
|---|---|
| `V_FMA_F32` | 128 |
| `V_PK_FMA_F16` / `V_DOT2_F32_F16` | 256 |
| **VOPD FP16 (`V_DUAL_DOT2ACC_F32_F16`)** | **512** |
| **WMMA FP16→FP32** | **512** |

WMMA **no eleva el techo arquitectónico**. Pero el pico **alcanzado** sí difiere:

| Ruta | TFLOPs | % de su pico |
|---|---|---|
| **rocWMMA, ≥2 bloques/CU** | **57.1** | **96 %** |
| VOPD `V_DUAL_DOT2ACC_F32_F16` | 43.8 | 73.8 % |
| `V_DOT2_F32_F16` single-issue | 28.7 | 96.7 % |

**WMMA entrega 1.30× más que la mejor ruta VOPD** y ~2× sobre dot2 single-issue.

Causa del techo de VOPD `[M]`: límite del **puerto de lectura SRC2** — FMAC/DOT2ACC
leen su acumulador como SRC2, lo que exige 6 lecturas VGPR/ciclo contra 4 bancos ×
3 puertos. `V_DUAL_FMAAK_F32` (addendo literal, sin SRC2) alcanza ~98 %.

Ciclos:

| Operación | Throughput |
|---|---|
| WMMA por CU (2 waves) | **16 cyc** (512 FLOPs/cyc) |
| WMMA por wave, bancos disjuntos | 32.0 cyc |
| WMMA por wave, todo en banco 0 (lo que genera HIPCC) | 34.0 cyc |
| WMMA INT4 | 8 cyc (2×) |
| `V_DOT2_F32_F16` | 1 cyc/instr, 2 MAC/lane/cyc |

**Bank conflicts de WMMA** `[M]`: banco = `vgpr_start % 4`. **A, B y C deben
estar los tres en bancos distintos**: separar solo un par da exactamente cero
mejora. El banco del destino es irrelevante; la acumulación in-place es 1 cyc/WMMA
más barata. **HIPCC siempre coloca los operandos en banco 0** → paga 34 en vez de
32 (6 %).

Se puede forzar con constraints de registro físico (`"{v[lo:hi]}"`) en inline
asm, **pero con cuidado**: hace falta un slot físico por fragmento vivo
simultáneamente. Un slot B compartido cuesta 28 `v_mov`/iteración; un slot A/B
único colapsa el prefetch a `s_waitcnt vmcnt(0)` entre WMMAs. Ambas variantes
ingenuas pierden mucho más que el 6 % que persiguen. **Tratar como tuning de
último recurso**, no como prioridad: es un 6 % sobre la parte WMMA de un kernel
que está 18-90× por debajo del ridge.

Notas: **SWMMAC no existe en gfx1151** (es gfx12) `[M]`. **No hay
`v_pk_fma_bf16`** en RDNA3.5 `[ISA]` → el path bf16 es 30–40 % más lento.

### 2.4 Camino a VOPD desde HIP `[M]`

**El requisito es que los `src0` de las dos instrucciones estén en BANCOS de VGPR
distintos** (`bank = idx % 4`). Registros distintos **no basta**:

| copias de `P` | bancos | pairing |
|---|---|---|
| `v20` / `v24` | 0 y 0 | **0 %** |
| `v20` / `v21` | 0 y 1 | **100 %** |

Aislado con registros fijados, sin cambiar nada más `[F7]`.

**Por qué falla en `P@V` por defecto:** el operando `P` es la probabilidad del
softmax difundida a `__half2`, y es **invariante en el bucle interno** sobre las
columnas de V. Todos los `fdot2` leen **el mismo VGPR** como `src0` — mismo
registro es trivialmente el mismo banco, así que **ningún par puede emparejarse,
ni con 2 acumuladores ni con 64**.

Visible en el ASM preexistente (`fleet/asm/out/q3b_dot2_dv256_sp1_st1.s`):
**47 de 51** `v_dot2acc_f32_f16` comparten `src0 = v96`. Los 17 `v_dual_dot2acc`
que aparecen no son dos dot2 emparejados: van con 17 `v_dual_mul_f32` — el
rescalado del softmax emparejado con un dot2. De ahí el 1.4 % que midió F2.

**La receta completa:**

| Requisito | Por qué |
|---|---|
| `__builtin_amdgcn_fdot2(a,b,acc,false)` | único camino desde HIP; inline asm no empareja |
| **≥2 copias de `P` en bancos distintos** | la condición real |
| `asm volatile("" : "+v"(x))` en las copias | sin esto CSE las refunde: 12 % vs 87 % |
| alternar `j & 1` | el pairer solo empareja instrucciones **adyacentes** |
| wave32 | VOPD es wave32-only `[ISA]` |

**No hay acantilado por número de acumuladores** `[F7]`: con 8/16/32 el
resultado es binario — 0 % con 1 copia de `P`, **100 % con 2**. La regla de
«≥2 acumuladores independientes» era una observación correcta con el **mecanismo
mal atribuido**, y por eso no transfirió.

Cuando el bucle ya tiene ≥4 operandos `P` genuinamente distintos
(`BLOCK_M=8`), llega al 100 % solo. El problema es específico del caso de `P`
difundido único.

### 2.5 LDS `[M]`

| Dato | Valor |
|---|---|
| Capacidad | **128 kB por WGP** (2 × 64 kB), máx 64 kB por work-group |
| Granularidad | 1024 B |
| Bancos visibles por wave | **32** → **128 B/ciclo por wave** |
| WGs residentes por WGP | `floor(128 kB / lds_per_wg)` en modo WGP |

Throughput y latencia:

| Op | cyc/op | B/cyc | Latencia |
|---|---|---|---|
| `ds_load_b32` | 1.0 | **128** | 32 cyc |
| `ds_load_b64` | 2.0 | **128** | 34 cyc |
| `ds_load_b128` | 6.0 | 85 | 38 cyc |
| `ds_store_b128` | 5.0 | **102** | ~34 |

Tres hallazgos que contradicen la práctica habitual:

1. **El ancho óptimo es opuesto para lecturas y escrituras**: leer con `b64`/`b32`
   (128 B/cyc), escribir con `b128` (102 B/cyc).
2. **Los bank conflicts NO cambian el throughput.** Un broadcast con conflicto
   32-way, un stream conflict-free y un pipeline manual dan el mismo cyc/op.
   Confirmado por 5 intentos fallidos de padding y XOR-swizzle en el epílogo de
   CK. **No invertir esfuerzo en swizzle anti-conflicto.**
3. **LDS es ~128 B/cyc por WGP y no escala con waves**: una sola wave ya satura.
   Más waves compran latency hiding, no ancho de banda de LDS.

Para `ds_read_b128` el factor de conflicto es `gcd(q,8)` con `q` = stride en quads
de 16 B; limpio ⟺ `q` impar. Padding: `P ≡ 16 − S (mod 32)`.

### 2.6 Cola de memoria `[M]`

| Contador | Capacidad |
|---|---|
| `VMcnt` (global) | **63** pendientes |
| `lgkmcnt` (LDS) | ~17 pendientes |
| Loads VMEM en vuelo **por wave** | **~10** (tope del VMEM return buffer) |

Emitir 63 loads y esperar una vez cuesta lo mismo que emitir 4 (~36 cyc/iter):
**el contador no es el cuello**. Emitir todas las cargas del prólogo y hacer
`s_waitcnt` selectivo es esencialmente gratis.

`s_waitcnt` diferencia contadores: `vmcnt` = global/buffer, `lgkmcnt` = LDS/SMEM.
Un `s_waitcnt 0` indiscriminado impide diagnosticar qué espera el kernel.

### 2.7 Presupuesto de ALU bajo la latencia de memoria `[M]`

Un kernel D1-pipelined esconde **~256 instrucciones VALU independientes** en la
ventana de latencia sin perder ancho de banda. **Con cadena dependiente, ~128**
(latencia VALU = 2 cyc, throughput 1 cyc).

Es el presupuesto para el online softmax por tile.

### 2.8 Frecuencia `[M]`

Boost ~2900 MHz; **sostenida bajo GEMM prolongado ~2100 MHz**. Las cifras por
ciclo medidas a boost sobreestiman ~27 % lo que verá una carga real.

El plateau limitado por DRAM es **independiente del reloj shader**; solo la región
rampa (pocas waves) escala con SCLK.

---

## 3. Split-KV: elegir NSEG

`NSEG` = segmentos en que se parte el eje KV. Cada uno produce `(m, l, acc)`
parciales; una reducción los combina.

### 3.1 Ocupación de partida (sin split-KV)

```
grid = (q_blocks, H_kv)      q_blocks = M // BLOCK_Q + 1
BLOCK_M = 16                 (fragmento de la unidad matricial)
BLOCK_Q = BLOCK_M // R       R = Hq / Hkv
```

| Caso | R | BLOCK_Q | filas útiles | WG | % de 40 CU | tiles KV |
|---|---|---|---|---|---|---|
| **Hq32/Hkv16, M=4, S=2048** | 2 | 8 | **8/16** | **16** | **20 %** | 128 |
| Llama3-8B, M=1, S=4096 | 4 | 4 | 4/16 | 8 | 10 % | 256 |
| Llama3-8B, M=4, S=4096 | 4 | 4 | 16/16 | 16 | 20 % | 256 |
| MHA (Hkv=32), M=1 | 1 | 16 | 1/16 | 32 | 40 % | 256 |

Dos observaciones:

- **La ocupación del tile y la del grid son problemas distintos.** El caso Llama
  M=4 llena el tile al 100 % y sigue en 20 % de la board.
- **Ningún caso realista pasa del 40 % sin split-KV.**

### 3.2 La tabla de decisión

Caso de estudio, KV = 32 MB:

| NSEG | WG | WG/WGP | WG/CU | waves/SIMD | tiles/seg | overhead |
|---|---|---|---|---|---|---|
| 1 | 16 | 0.8 | 0.4 | 0.8 | 128 | 0 % |
| 2 | 32 | 1.6 | 0.8 | 1.6 | 64 | 1 % |
| 3 | 48 | 2.4 | 1.2 | 2.4 | 43 | 2 % |
| **5** | **80** | **4.0** | **2.0** | **4.0** | 26 | **2 %** |
| 8 | 128 | 6.4 | 3.2 | 6.4 | 16 | 3 % |

(waves/SIMD con 4 waves por WG.)

Criterios, en orden:

1. **Cobertura mínima**: `NSEG = ceil(40 / WG_base)` para que ningún CU quede
   vacío. Aquí NSEG=3 (NSEG=2 cubre los 20 WGP).
2. **Suelo de waves en vuelo**: 2–4 waves/SIMD para esconder latencia (§2.2).
   **Este requisito domina sobre el de cobertura.**
3. **Alineación**: `WG` múltiplo de 20 **y** 40 — evita la cola de la última
   ronda.
4. **Wave quantization**: `Hq × NSEG` múltiplo de 20.
5. **Techo**: overhead de parciales lineal con NSEG; hasta ~3 % es ruido.
6. **Suelo**: ≥ 8 tiles por segmento.

**NSEG = 5** satisface los seis. Confirmado por tres vías independientes:

| Criterio | NSEG=5 |
|---|---|
| Alineación de grid | 80 = 4×20 = 2×40 ✓ |
| Waves en vuelo | 4.0/SIMD — en el óptimo 2–4 ✓ |
| Wave quantization | `32 × 5 = 160 = 8 × 20` ✓ |

Alineación exacta a 20 y 40 solo ocurre en **NSEG ∈ {5, 10, 15}**.

Utilización con efecto cola (`WG / (P × ceil(WG/P))`):

| NSEG | WG | P=20 (WGP) | P=40 (CU) |
|---|---|---|---|
| 2 | 32 | 80 % | 80 % |
| 3 | 48 | 80 % | 60 % |
| 4 | 64 | 80 % | 80 % |
| **5** | **80** | **100 %** | **100 %** |
| 8 | 128 | 91 % | 80 % |

NSEG=5 da 100 % tanto si el limitante son los WGP como si son las CU — no hace
falta resolver cuál manda.

### 3.3 Coste de los parciales

```
bytes = NSEG × M × H_q × (D × 4 + 8)        # acc fp32 + (m, l)
```

Con NSEG=5: 0.66 MB sobre 32 MB de KV = **2 %**. Se escriben en HBM y se releen
en la reducción.

### 3.4 Reparto real de tiles

Tiles totales = `S / TILE = 2048/16 = 128`. El reparto usa `ceil`:

```
tiles_per_segment = cdiv(S, NSEG × TILE) = cdiv(2048, 80) = 26
NSEG=5 → [26, 26, 26, 26, 24]      suma = 128
```

El WG crítico hace 26 frente a un ideal de 25.6: **desbalance 1.6 %**.

Más relevante que el desbalance: con contextos cortos `ceil` reparte de más al
principio y los últimos segmentos **quedan vacíos**:

| S | NSEG | reparto | activos |
|---|---|---|---|
| 2048 | 5 | `[26,26,26,26,24]` | 5/5 |
| 512 | 16 | `[2 × 16]` | 16/16 |
| **300** | **8** | `[3,3,3,3,3,3,1,0]` | **7/8** |
| **100** | **16** | `[1×7, 0×9]` | **7/16** |

Condición para que los NSEG segmentos tengan trabajo:

```
S > (NSEG - 1) × tiles_per_segment × TILE
```

### 3.5 NSEG no puede depender de S

`S` crece en cada paso y con CUDA graphs el grid se captura: **NSEG debe ser
estático**, fijado por propiedades del modelo y la board.

El daño es asimétrico y el suelo es bajo:

| S | NSEG=5 | NSEG=8 | NSEG=16 |
|---|---|---|---|
| 128 | 4/5 | 8/8 | **8/16** |
| 256 | 4/5 | 8/8 | 16/16 |
| 512+ | 5/5 | 8/8 | 16/16 |

Al **crecer** S no pasa nada malo: solo suben los tiles por segmento. Los
segmentos vacíos solo aparecen por debajo de S ≈ `NSEG × TILE × 2` (unos 160
tokens con NSEG=5), y en decode S solo crece.

Para cubrir el caso corto sin tocar el grid basta con que los WG sin trabajo
hagan early-return.

---

## 4. Taxonomía de ejes y orden del grid

### 4.1 Los índices de la contracción

```
O[m, h, e] = Σ_s  P[h, m, s] · V[s, h, e]
P[h, m, s] = softmax_s( Σ_d Q[m,h,d] · K[s,h,d] / √d )
```

| Índice | Tamaño | Tipo | Combinador | ¿Usado? |
|---|---|---|---|---|
| `m` | M | libre | — | sí (BLOCK_Q) |
| `r` | R = n_rep | libre | — | sí (filas del tile) |
| `kv` | H_kv | libre | — | sí (dimensión del grid) |
| **`e`** | **D_v** | **libre** | **concatenación** | **no** |
| `s` | S | reducción | monoide `(m,l,acc)` | sí (split-KV) |
| `d` | D_qk | reducción | suma | no (interno al producto) |

**Regla general: un índice libre se parte por concatenación (gratis en
correctitud); uno de reducción necesita un combinador.** Si el combinador es caro
o el índice obliga a releer operandos, la partición deja de compensar aunque sea
matemáticamente válida.

**`e` (D_v) es el único eje libre sin usar.** No aparece en `P`, solo en `V` y en
la salida — partirlo es concatenación pura, verificado con error **exactamente
0.0** `[D]`:

```python
o1 = einsum("hms,shd->mhd", P, V[..., :half])
o2 = einsum("hms,shd->mhd", P, V[..., half:])
cat(o1, o2) == ref      # exacto, sin combinador
```

**Como eje de grid no sirve**: cada trozo necesita `P` completo, luego relee K
(1.5× de tráfico con 2 trozos, 2.5× con 4). Inaceptable en memory bound.

**Como partición interna al WG sí**: `P` se calcula una vez y se reutiliza en las
dos pasadas, sin releer nada. Reduce a la mitad los bytes vivos de `acc` (§6).

Ejes descartados: **capas** (libre pero serializado por la dependencia del
modelo) y **`d`/D_qk** (split-K clásico; solo compensa si `S × M` fuera
minúsculo).

### 4.2 Orden del grid: localidad de caché

El `program_id` que varía más rápido decide qué WGs se despachan juntos y por
tanto qué comparten caché. Precedente: `USE_SWAPPED_GRID`
(`triton_unified_attention.py:318`) intercambia dos dimensiones en gfx1151 por
rendimiento medido.

Con NSEG dimensionado por ocupación el segmento ya es menor que L2, así que la
localidad sale gratis. Lo que decide el orden es **cuánta KV está viva a la vez**,
y eso lo fija el eje **lento**:

| Eje lento | KV residente (NSEG=5, Hkv=16, S=2048) | Nivel |
|---|---|---|
| **H_kv** (1 kv head viva) | **0.40 MB** | **L2** |
| NSEG (16 kv heads vivas) | 6 MB | solo MALL |

Orden propuesto:

```
rápido → H_q     (los R WGs del grupo comparten kv head → reuso inmediato)
medio  → NSEG    (segmentos de la misma kv head)
lento  → H_kv    (una sola kv head viva → working set en L2)
```

Compensa porque el KV de una capa son **32 MB, justo el límite de MALL**: bajar
el working set concurrente de 6 MB a 0.40 MB deja MALL para prefetch en vez de
como almacén.

Dimensionado desde la caché, como **cota superior** del segmento:

```
S_cache = cache_bytes / (D × 2 × elem_size)
```

Con D=256 fp16: `S_cache(L2 2 MB) = 2048`, `S_cache(MALL 32 MB) = 32768` `[D]`.
Con S=2048 el KV de una kv head ya cabe en L2, así que este criterio no restringe
— manda la ocupación.

**Caveat:** asume que el despachador emite WGs en orden creciente de ID
linealizado. Es lo habitual pero **no está garantizado**. El experimento barato
es probar ambos órdenes.

### 4.3 El eje H_q como paralelismo adicional

Usar las query heads de un grupo GQA como eje de WG. No compensa como sustituto
de split-KV, por tres razones:

1. **Aporta solo `R`×** (2–8). En el caso de estudio: 16 → 32 WG, que cubre los
   20 WGP con cola y no llega a 40 CU limpio.
2. **Rompe el tile.** Las `R` query heads **son las filas del fragmento
   matricial**; separarlas baja `BLOCK_M` de 8 a 4. La condición para no bajar de
   16 filas es `chunk ≥ ceil(16/M)`, que con R=2 es imposible.
3. **Peor tráfico.** Split-KV lee segmentos **disjuntos** (0 relecturas) y reduce
   la huella por WG; H_q-split relee la misma KV `R` veces.

Además, recorrer grupos en secuencia **serializa lo que se quiere paralelizar**:
si los grupos están vivos a la vez, el KV residente es el de todas las kv heads
concurrentes (32 MB), no el de una (2 MB).

**Útil solo como segundo eje con `R ≥ 8`** (MQA / GQA extremo), donde H_kv es
pequeño y NSEG por sí solo no llena la board.

| Eje | Factor | Coste |
|---|---|---|
| **KV (NSEG)** | libre (5–32) | parciales en HBM (~2–6 %) |
| H_q | R (2–8) | rompe el tile si `R < ceil(16/M)×2`; relee KV |
| M | 1–5 | lo fija el draft, no el kernel |
| H_kv | fijo | propiedad del modelo |

---

## 5. Unidad matricial: WMMA vs dot2

Decisión de **segundo orden** (§1.2), pero con consecuencias en registros.

### 5.1 Dónde se paga el padding

`BLOCK_M = 16` viene del fragmento WMMA 16×16×16, no del problema. Con dot2 no
existe esa restricción: `BLOCK_M = M × R`.

En `Q@Kᵀ` el único eje con padding es M: `N = TILE = 16` siempre lleno, `K = D`
se encadena (D=256 → 16 fragmentos).

| M | R | M×R | fragmentos | filas | eficiencia |
|---|---|---|---|---|---|
| 4 | 4 | 16 | 1 | 16/16 | **100 %** |
| 2 | 8 | 16 | 1 | 16/16 | **100 %** |
| 8 | 2 | 16 | 1 | 16/16 | **100 %** |
| 4 | 8 | 32 | 2 | 32/32 | **100 %** |
| 3 | 4 | 12 | 1 | 12/16 | 75 % |
| **4** | **2** | **8** | 1 | **8/16** | **50 %** |
| 5 | 4 | 20 | 2 | 20/32 | 62 % |
| 1 | 4 | 4 | 1 | 4/16 | 25 % |

`M × R ≡ 0 (mod 16)` equivale a **M múltiplo de `BLOCK_Q`**, porque
`BLOCK_Q = 16/R`.

### 5.2 El padding cuesta FLOPs **y** registros

Los dos productos tienen costes asimétricos:

| Producto | Salida | Coste del padding |
|---|---|---|
| `Q @ Kᵀ` | `S[16, TILE]` = **1 KB** | solo FLOPs |
| `P @ V` | `acc[16, D_v]` = **16 KB** | **FLOPs + VGPR, vivos todo el bucle** |

Solo el segundo fija el tamaño de `acc`. Con M=4/R=2, **la mitad de `acc` es
padding**: 8 KB de 16 KB.

### 5.3 Veredicto: dot2/VOPD para `P@V` `[M, F7]`

Medido con el pairing arreglado (D_v=256, BLOCK_M=8, wave32, normalizado a MACs
útiles):

| Ruta | cyc/iter | vs WMMA | VGPR |
|---|---:|---:|---:|
| WMMA | 640.9 | 1.00× | 153 |
| dot2/VOPD | 434.4 | 1.48× | 75 |
| **dot2/VOPD, 2 copias de `P`** | **407.8** | **1.57×** | **83** |
| `v_pk_fma_f16` | 647.6 | 0.99× | 50 |

El 1.57× concuerda con la teoría: el desperdicio de padding de WMMA (8 de 16
filas vacías con `M×R=8`) escalado por el límite del puerto SRC2 predice 1.54×.
`v_pk_fma_f16` **no es competitivo**: empata con WMMA pese a usar 50 VGPR.

**Lo que decide no es el punto, es la pendiente de registros.** Barriendo
operandos de V residentes: WMMA crece **+6 VGPR** por operando y **desborda a
249 VGPR con 24 spills** con V totalmente residente; dot2 crece **+0.9** y nunca
desborda (96 VGPR). Es una pendiente **6.7× peor**.

**Encuadre honesto:** el kernel es 18–90× memory-bound, así que **el 1.57 % de
VALU no moverá el wall time**. Importa porque **153 → 83 VGPR compra ocupación y
profundidad de prefetch**, que es lo que sí ataca el cuello de latencia de DRAM.
Medido desde el otro lado: dot2 sostiene `NSTAGE=8` sin spill, WMMA se para en 4.

### 5.4 Elegir M en spec decoding

```
M óptimo = múltiplo de BLOCK_Q = 16 / R
  R=2  →  M ∈ {8, 16}
  R=4  →  M ∈ {4, 8}
  R=8  →  M ∈ {2, 4, 6}
```

M=5 con R=4 es el peor punto de la zona útil (62 % y el doble de fragmentos).
Entre dos M con tasa de aceptación similar, elegir el múltiplo.

**Ojo con la división entera**: `BLOCK_Q = 16 // R` descarta el resto, así que si
R no divide a 16 se pierden filas **permanentemente**, con independencia de M
(R=7 → 14/16; R=6 → 12/16).

---

## 6. Registros y tamaño de wave

### 6.1 El consumidor dominante

`acc[BLOCK_M, D_v]` en fp32 = `BLOCK_M × D_v × 4` bytes por WG. **Es fijo**; lo
que varía es cómo se reparte:

| Estrategia | acc vivo | VGPR/lane | ¿relee K? |
|---|---|---|---|
| WMMA, `D_v` entero, wave32 | 16 KB | **128** | no |
| WMMA, `D_v` entero, wave64 | 16 KB | 64 | no |
| WMMA + `D_v`/2, wave32 | 8 KB | 64 | no |
| dot2, `D_v` entero, wave32 | 8 KB | 64 | no |
| dot2 + `D_v`/2, wave32 | 4 KB | **32** | no |
| `D_v` como eje de **grid** | 8 KB | 64 | **sí, 2×** |

**Las tres palancas son ortogonales y se componen**: tamaño de wave (÷2), partir
`D_v` internamente (÷2), dot2 en `P@V` (BLOCK_M de 16 a M×R).

Con 128 VGPR/lane la ocupación cae a 10 waves/SIMD; con 64 se alcanza el tope de
16. **Objetivo: ≤ 96 VGPR/lane**, dejando margen para Q, direcciones, block table
y temporales.

### 6.2 Desambiguación de «D»

| | Qué es | Papel |
|---|---|---|
| **D_qk** | eje de **reducción** de `Q@Kᵀ` | la «K» del fragmento; se recorre y acumula |
| **D_v** | eje **libre** de `P@V` | la «N» del 2º producto — **el ancho de `acc`** |
| **wave size** | reparto de `acc` entre lanes | no cambia el tamaño de `acc` |

### 6.3 wave32 vs wave64

**Los bytes en vuelo no dependen del tamaño de wave:**

```
bytes en vuelo = TILE × D × 2 × stages
```

Por la ley de Little lo que oculta latencia son **bytes**, no instrucciones.
wave32 emite 32 instrucciones de 512 B donde wave64 emite 16 de 1024 B: los
mismos 16 KB. LDS y bytes en vuelo son idénticos.

Lo que sí cambia:

| D | wave32 VGPR/lane | wave64 VGPR/lane |
|---|---|---|
| 64 | 40 | **20** |
| **256** | **136** | **68** |

(`acc[16,D]` + `S[16,TILE]`, ambos fp32, con WMMA.)

Datos medidos que deciden:

- **WMMA no tiene penalización en wave64** `[M]`: 34.5 cyc/WMMA y 237–238
  FLOPs/cyc/CU idénticos. En wave64 se emiten 2 pases VALU, pero los lanes 32–63
  llevan A/B replicados — un solo fragmento, no dos.
- **Para saturar DRAM, wave32 y wave64 convergen** `[M]`: una vez saturado el
  DDR (≥128 waves) ambos dan 215–229 GiB/s (ratio 0.98–1.06). En la región rampa
  wave64 da ~2×, pero esa región no es donde opera un kernel bien dimensionado.
- **VOPD es wave32-only** `[ISA]`.

**Decisión: wave32.** wave64 no aporta ancho de banda en el régimen saturado y
excluye VOPD; su única ventaja (la mitad de VGPR/lane) se consigue igual
partiendo `D_v` o usando dot2 en `P@V`.

**Trampa documentada** si se prueba wave64 `[M]`: `v_add_co_u32` debe usar `vcc`
(64 bits), no `vcc_lo`. Recompilar ingenuamente **calcula direcciones mal en
silencio** para las lanes 32–63 y produce un número plausible pero falso.

### 6.4 Efecto de D en la carga

wave32, `dwordx4` = 16 B/lane = 512 B por instrucción:

| D | B/fila | lanes/fila | filas por instr | instr por TILE=16 |
|---|---|---|---|---|
| **64** | 128 | 8 | 4 | **4** |
| 128 | 256 | 16 | 2 | 8 |
| 256 | 512 | 32 | 1 | 16 |

Las lanes por fila dependen solo de `D × 2 / 16` y no cambian con el tamaño de
wave.

Presupuesto de VGPR/lane (máx. 256, con WMMA):

| D | TILE | acc | S[16,TILE] | total |
|---|---|---|---|---|
| 64 | 16 | 32 | 8 | **40** |
| 64 | 32 | 32 | 16 | 48 |
| **256** | 16 | **128** | 8 | **136** |

Con D=64 sobra presupuesto y el cuello potencial es el número de instrucciones
independientes para el issue; con D=256 el cuello son los registros.

Regla para D pequeño: `TILE ≈ 512/D` acotado a ≥ 16. Precedente:
`_get_tile_size` sube TILE a 32 para Gemma3 y fp8.

**Swizzle con paged KV**: 8 lanes consecutivas leen una fila con D=64. Si las
filas de un tile caen en bloques físicos distintos, cada instrucción se fragmenta
en varias transacciones. El mapeo `lane → (fila, offset)` debe mantener cada
grupo de 8 lanes en la misma línea de caché.

---

## 7. Memoria: las palancas de mayor impacto

### 7.1 Conflicto de sets de caché: el layout, no el padding

**Mecanismo CERRADO `[M]`: conflicto de sets en caché, no contención de canal
DDR.** Prueba discriminante con bytes constantes: con **1 sola kv-head viva el
efecto desaparece** (63.7 vs 63.5 GiB/s), aparece con 2 y satura en 4. Un efecto
de canal DDR no puede depender de eso. Los contadores lo cierran: `FETCH_SIZE`
(31.25 MiB) y GL2C hit (0.04 %) **idénticos** entre caso rápido y acantilado,
mientras `GL1C_STALL_GL2_GL1` va 177 → **1294** → 119.

### La solución es cambiar el orden de dimensiones, no añadir bytes

`NHD` pone las kv-heads contiguas dentro del token: **1 KiB entre heads
consecutivas**, así que las `H_kv` compiten por los mismos sets. `HND` da a cada
head un bloque contiguo de **16 KiB**, dispersándolas.

**`HND` es una permutación pura: `page_size_bytes` idéntico** (256 KiB,
verificado en código). Cero bloques perdidos, cero pérdida de concurrencia.

| Medida `[M, F5]` | Resultado |
|---|---|
| Kernel Triton de producción | **1.090×** (282 → 259 µs) |
| Batch 16 / 64 | **1.16× / 1.12×** — gana *más* en batch alto |
| Prefill | 1.09× |
| `reshape_and_cache` | sin degradar (19.34 vs 19.35 µs) |
| Regímenes probados | gana en **los 8** |

**Predicción falsable confirmada**: si el mecanismo es competencia entre
kv-heads, debe escalar con `H_kv` y desaparecer con `H_kv=1`. Se cumple:
H_kv=8/16 → 1.06–1.13×; **H_kv=1 → 1.00×**.

### El padding queda descartado

- **No transfiere al layout paginado real**: el 1.07× del harness denso baja a
  1.04×, y **no es monótono** — solo `KV_PAD=64` ayuda; 8, 16, 32 y 128 son
  *peores* que 0.
- **No existe un "padding mínimo barato"**: `KV_PAD=8` (+1.6 % de memoria) da
  **−2.3 % de rendimiento**.
- **Cuesta bloques**: `KV_PAD=64` son −11 % de VRAM útil para ganar 4 %, cuando
  HND da 9 % gratis.
- **No se combinan**: HND+pad (159 µs) es *peor* que HND solo (152 µs).

### El acantilado de paso de fila: real, y hay un guard barato

**El predictor es el PASO DE FILA** (`2 * head_size * esz`, la extensión de la
última dimensión), **no `stride_head`** `[F8]`.

Prueba por reducción al absurdo: bajo HND el `stride_head` es `block_size` veces
el paso de fila, luego **múltiplo de 2048 para todo `hs ≥ 64`**. Si el predictor
fuese `stride_head`, HND estaría siempre en el acantilado — pero HND mide **más
rápido** (150.2 vs 161.4 µs, §0.1). Verificado.

Bajo NHD el paso de fila *coincide* con `stride_head`; bajo HND coincide con
`stride_token`. De ahí la confusión inicial.

**Magnitud real: 1.09× (NHD) / 1.32× (HND)**, medido end-to-end en vLLM con
`D=512`, no el 1.90× del microbenchmark. La diferencia: aquel padeaba un D=256
hasta un paso de 2048, dejando **un agujero de 1 KiB por fila**; con D=512 el
paso de 2 KiB está **lleno**. Con agujero el acantilado sí es grande (D=256
padeado a paso 4096: 348 → **587** → 349 µs).

### ⚠️ Padear mal es peor que no padear

| pad | µs | vs base | `pad % 32` |
|---|---|---|---|
| 0 (base) | 1167 | 1.00× | — |
| **+8** | **1712** | **1.47× PEOR** | 8 |
| **+16** | **1709** | **1.46× PEOR** | 16 |
| **+48** | **1702** | **1.46× PEOR** | 16 |
| +32 / +64 / +256 | bien | — | 0 |

Los tres malos son exactamente los **no múltiplos de 32 B**. Un guard que
"desalinee un poco" con aritmética ingenua puede aterrizar en 48 B y **empeorar
1.47×**. **El padding debe ser múltiplo de 32 B.**

### El guard

```
si  paso_de_fila % 2048 == 0  y  num_kv_heads >= 16:
        padear +32 B
```

- **+32 B captura el 82 % de la ganancia** (1.22× de 1.34×) por **1.6 % de
  memoria**, frente al 12.5 % que costaría +256 B. Ocho veces más barato.
- **Condicionado a `H_kv ≥ 16`** porque el efecto escala con las kv-heads:
  1 → 1.004×, 2 → 1.029×, 4 → 1.042×, 8 → 0.99×, **16 → 1.09×**. Así
  Llama/Qwen/Mistral no pagan un byte.
- **Vale en ambos layouts** y rinde más bajo HND (1.32×). HND **no** arregla el
  acantilado: son ortogonales — HND ataca el `stride_head`, el guard el paso de
  fila.
- **Nunca aplicarlo en general**: en D=128 cuesta 1.10×, en D=256 cuesta 1.24×.

**Config real afectada: Gemma4** (`transformers_utils/configs/gemma4.py:27`,
`global_head_dim = 512` en capas full_attention → paso de 2048 exacto). DeepSeek
D=512 va por MLA, otro backend.

**Coste de implementación**: la condición es una línea, pero
`kv_cache.split(hs, dim=-1)` (`triton_attn.py:687`) asume que la última dim es
exactamente `2*hs`. Hace falta el mismo recorte de vista que ya usa la rama de
cuantización per-token-head. **Ese es el trabajo real.**

### Qué tocar en vLLM

**Una línea**: el default `"NHD"` en
`vllm/distributed/kv_transfer/kv_connector/utils.py:51`, gateado por
`on_gfx1151()`. No hace falta tocar `get_kv_cache_shape`,
`get_kv_cache_stride_order` (ya soporta HND) ni `block_size` (palanca errática:
ningún valor gana en todas las formas).

HND ya lo exige NIXL y lo asume FlashInfer, así que no es un camino exótico.

**Cautelas antes de fijar el default** `[?]`: sin probar con
`kv_cache_dtype=fp8` (cambia `padded_hs` y por tanto los strides); con
`H_kv ≤ 4` HND empata en vez de ganar.

### 7.2 Profundidad de pipeline

`[M]` `D` = `global_load_b128` en vuelo por wave. % del pico de 238.4 GiB/s:

| waves (w/SIMD) | D1 | D2 | D4 | D8 | D12 |
|---|---|---|---|---|---|
| 16 (0.2) | 11 % | 22 % | 43 % | 77 % | **92 %** |
| 32 (0.4) | 22 % | 41 % | 70 % | **91 %** | 95 % |
| 80 (1.0) | 49 % | 78 % | **94 %** | 95 % | 95 % |
| 160 (2.0) | 79 % | **94 %** | 95 % | 96 % | 96 % |

- **Tope de ~10 loads en vuelo por wave** (VMEM return buffer). De D2 a D10
  escala lineal (+1.7 GiB/s por nivel); D11+ no aporta.
- **Regla de oro: 4–6 `load_b128` en vuelo por SIMD** → 94–97 %.
- **32 threads por bloque (1 wave) es óptimo**; más threads empeora ligeramente.
- El default del compilador (`s_waitcnt vmcnt(0)` tras cada load) da **0.7 %**
  del pico con 1 wave.
- Coste: **4 VGPR por nivel**.
- **El siguiente `global_load` debe ir inmediatamente después del `s_waitcnt`**;
  meter ALU de cálculo de dirección en medio cuesta **10 %**.

Aplicado: con NSEG=5 (80 WG = 1.0 wave/SIMD) hace falta **D≈6**; con NSEG=10
(2.0 wave/SIMD), D≈3.

Little para DRAM: 545 ns × 232 GiB/s ≈ **17.4 KiB en vuelo** por CU `[D]`.

### 7.3 Coalescencia

`[M]` BW máximo según el patrón de cada `load_b128`:

| Patrón | BW máx |
|---|---|
| 2 filas × 256 B contiguos | **225 GiB/s** |
| 4 filas × 128 B contiguos | 214 GiB/s |
| 8 filas × 64 B contiguos | 198 GiB/s |
| 16 filas × 32 B contiguos | 160 GiB/s (200 con D≥10) |

Con D=256 fp16 una fila son **512 B contiguos** → mejor caso, siempre que el
paged KV no fragmente el tile (§6.4).

Con accesos no coalescidos **pasarse de 4–6 loads en vuelo penaliza
activamente** (hasta <100 GiB/s).

### 7.4 LDS: qué hacer

De §2.5:

- **Leer con `ds_load_b64`** (128 B/cyc), no `b128` (85 B/cyc).
- **Escribir con `ds_store_b128`** (102 B/cyc).
- **No invertir esfuerzo en swizzle anti-conflicto** — no afecta al throughput.
- Más waves no dan más BW de LDS; para eso hacen falta más WGPs.

### 7.5 Staggering

`[M]` Los GEMM que alcanzan el pico usan `StaggerU` con `SUS=256`: cada workgroup
empieza su recorrido en un offset distinto, y 256 B *"coincide con la granularidad
de interleave de canal LPDDR5x en gfx1151"*.

Aplicable a split-KV: **escalonar el offset de inicio de KV por workgroup en
múltiplos de 256 B**.

### 7.6 Colocación determinista de workgroups

`[M]` Estable entre runs, sin leer `HW_ID1`:

```
k   = blockIdx.x
SE  = (k & 1) ? 0 : 1
SA  = (k / 2) / 5
GL1 = SE*2 + SA          // 0..3, cuál de los 4 GL1 de 256 KB
wgp = (k / 2) % 5        // 0..4
```

Permite colocar en el mismo SA las query heads que comparten kv head, para
reutilizar KV en GL1 (hit ~50 ns vs ~70 ns de GL2).

---

## 8. Precedentes en el stack

### 8.1 vLLM: el path 3D ya cubre spec decoding

La rama **`rogarcia.gfx1151-3d-attn-tuning`** (ROCm/vLLM, commit `77698ec`)
habilita el kernel 3D para MTP/spec decoding. El gating por forma del batch
(`max_seqlen_q > 1`) se sustituye por una **comprobación de capacidad**
(`triton_unified_attention.py:1165`):

```python
use_3d = not (
    seq_threshold_3D is None
    or ...
    or softmax_segm_output.shape[0] < q.shape[0]
    or softmax_segm_max.shape[0] < q.shape[0]
    or softmax_segm_expsum.shape[0] < q.shape[0]
    or num_seqs > seq_threshold_3D
    or is_batch_invariant
)
```

Con los buffers dimensionados por **tokens**, no por secuencias
(`triton_attn.py:203-208`):

```python
spec = vllm_config.speculative_config
max_query_len_3D = 1
if spec is not None and spec.num_speculative_tokens is not None:
    max_query_len_3D += spec.num_speculative_tokens
segm_rows = self.seq_threshold_3D * max_query_len_3D
```

Los buffers `softmax_segm_*` se indexan por índice absoluto de token, así que el
path 3D necesita una fila por token de query. Dos propiedades del diseño:

- **Spec decoding entra en 3D**: el verify step lleva `1 + num_speculative_tokens`
  por secuencia y los buffers están dimensionados para eso.
- **Prefill cae a 2D por construcción**, sin condición explícita: su número de
  tokens no está previsto en el dimensionado, así que la comprobación falla sola.

Reutilizable para el kernel HIP: **dimensionar los parciales por tokens** y
**gatear por capacidad** en vez de por forma del batch, para que los casos no
previstos degraden solos a la ruta segura.

Lo que no resuelve: los parámetros siguen siendo los de Triton (NSEG=16, TILE=16,
BLOCK_M=16). Nada de §3, §4.2, §6 ni §7 está aplicado ahí — en particular el
padding del stride y la profundidad de pipeline. **La rama desbloquea el camino;
la optimización sigue pendiente.**

### 8.2 wvSplitK: el análogo estructural de decode

Kernel skinny de vLLM para batch 1–8, el precedente más cercano a nuestro caso:

- `grid = dim3(CuCount)` = **20 WG persistentes, uno por WGP**
- `block = dim3(32, WvPrGrp)` — 32 threads = 1 wave
- **Instancias compiladas por separado para N = 1..5** (constante de compilación)
- **Activaciones a LDS; pesos directos desde global con non-temporal, bypassing
  L1** → traducido: **Q a LDS, K/V streaming non-temporal**
- Reducción cross-lane con **DPP `row_shr` + `wave_shr`**, solo el último lane
  escribe — el patrón para el max/sum del online softmax

Nota de API: **`num_compute_units()` devuelve WGPs, no CUs** (20 en Strix Halo,
que tiene 40 CU).

### 8.3 Split-K y Stream-K en RDNA

**Stream-K es una técnica de descomposición de trabajo, no una feature de
hardware.** Es implementable en RDNA3.5: todas las primitivas existen `[ISA]` —
grid persistente, `GLOBAL_ATOMIC_ADD_F32`, `GLOBAL_ATOMIC_CMPSWAP_F32`,
`GLOBAL_ATOMIC_CSUB_U32`, atómicos device-scope y `VMcnt`/`VScnt` para ordenar.

Lo que sí dice el dato medido: **en las soluciones shipped de hipBLASLt, StreamK
es exclusivo de CDNA** — todas las soluciones RDNA shipean `StreamK: 0`. No hay
parámetros de referencia que copiar; hay que caracterizar de cero.

Cuando CDNA lo usa, shipea `StreamKAtomic=0`: **parciales a workspace + fixup
determinista, nunca atómicos**. Es una elección de reproducibilidad, y una buena
guía por defecto para nuestra reducción (parciales fp32 + `(m,l)` en workspace).

Nuestro esquema con NSEG fijo es equivalente a **GlobalSplitU**, no a Stream-K:

| | NSEG fijo | Stream-K |
|---|---|---|
| Reparto | NSEG segmentos iguales por (head, tile-Q) | chunks iguales sobre grid persistente |
| Cola de la última ronda | posible (§3.2) | eliminada por construcción |
| Segmentos vacíos con S corto | sí (§3.4) | no |
| Complejidad | baja | fixup de tiles parciales |

Stream-K resolvería de raíz los dos problemas que §3 trata a mano. **Merece
evaluarse**, sobre todo si el kernel debe servir un rango amplio de S.

Contrapunto sobre la magnitud: con `num_seqs=1` el espacio de iteración es
`Hq × ceil(S/TILE) = 32 × 128 = 4096` chunks sobre 20 WGP = 204.8 por WGP `[D]`.
La cuantización de un reparto fijo ya es del 0.4 %, así que la ganancia aquí
sería menor que en GEMM — pero crece cuando S es corto.

---

## 9. Implicaciones de escribir el kernel en HIP

### Lo que gana relevancia

- **Control explícito del solapamiento**: `s_setprio`, `sched_group_barrier`,
  `s_waitcnt` manual. El entrelazado carga/cómputo pasa a ser decisión propia.
- ~~`global_load_lds` (global → LDS directo)~~ **NO EXISTE en gfx1151** `[M]`.
  Es CDNA-only: `llvm-mc` lo rechaza en gfx1151/gfx1100/gfx1200 y lo acepta en
  gfx942; el builtin de HIP nombra la target-feature ausente
  (`vmem-to-lds-load-insts`). Global→LDS **tiene que pasar por VGPRs**.
- **Colocación de operandos WMMA en bancos distintos**: 6 % que HIPCC deja sobre
  la mesa (§2.3).
- **`__builtin_amdgcn_fdot2`** si se usa la ruta VOPD (§2.4).
- **Swizzle de LDS explícito** para el caso paged (§6.4).
- **Layout de `acc` en registros**: controlar el mapeo del fragmento permite
  fusionar el epílogo del online softmax sin round-trip por LDS.

### Lo que deja de ser restricción

`_get_tile_size`, `num_stages`, `BLOCK_M`, `USE_SWAPPED_GRID` son parámetros de
la implementación Triton. En HIP se eligen libremente: pasan de restricción a
variable de diseño.

Se conserva: la taxonomía de ejes (§4.1, es matemática), el dimensionado de NSEG
(§3), la aritmética de VGPR y LDS (§6), y el orden de ejes del grid (§4.2, en HIP
se controla mejor desde un `blockIdx` linealizado).

### Medición

**Cronometrar con el realtime clock, no `SHADER_CYCLES`** `[M]`: se apaga durante
stalls de L2+ y hace wrap a 2²⁰. Un acceso a L2 cuenta solo ~33 ciclos activos en
~100 ns.

**Artefacto de benchmark**: con S=2048 el KV son 32 MB y **puede quedarse en MALL
entre iteraciones**, dando números falsamente buenos. Rotar buffers.

---

## 10. Checklist

Dada `(Hq, Hkv, M, D, S)` y el hardware:

**Reparto del trabajo**

1. `R = Hq/Hkv` → `BLOCK_Q = 16/R` → `q_blocks = M//BLOCK_Q + 1`
2. `WG_base = q_blocks × Hkv` → ¿cubre los 20 WGP / 40 CU? (§3.1)
3. Si no: `NSEG = ceil(40 / WG_base)` como suelo; subir hasta 2–4 waves/SIMD
4. Preferir NSEG que alinee con 20 **y** 40, y que haga `Hq × NSEG` múltiplo de
   20 (§3.2)
5. Verificar: overhead de parciales ≤ 3 %, tiles/segmento ≥ 8, sin segmentos
   vacíos para el S mínimo esperado (§3.4, §3.5)
6. Orden de ejes: `H_q` rápido → NSEG → `H_kv` lento (§4.2)

**Unidad matricial**

7. `M × R ≡ 0 (mod 16)` → WMMA en ambos productos. Si < 16 → WMMA en `Q@Kᵀ` y
   medir `P@V` (§5.3)
8. Comprobar filas útiles: `BLOCK_Q × R` pierde resto si R no divide a 16 (§5.4)

**Registros** — el que más aprieta con D grande

9. `acc[BLOCK_M, D_v]` fp32 = `BLOCK_M × D_v × 4` B → `/wave` = VGPR/lane
10. Palancas componibles: wave size (÷2), partir `D_v` internamente (÷2), dot2 en
    `P@V` (§6.1)
11. **Objetivo ≤ 96 VGPR/lane** para no perder ocupación (§2.2)

**Memoria**

12. **Padear el stride de head/página fuera de potencia de 2** (§7.1)
13. Profundidad de pipeline: 4–6 loads en vuelo por SIMD, máx ~10 por wave (§7.2)
14. `TILE`: bytes/tile = `TILE × D × 2`; subir TILE si D es pequeño (§6.4)
15. K/V non-temporal bypassing L1; Q a LDS (§8.2)
16. Lecturas de LDS con `b64`, escrituras con `b128` (§7.4)
17. Escalonar el offset de inicio de KV por WG en múltiplos de 256 B (§7.5)

---

## 11. Estado y trabajo restante

Todo lo que el usuario pidió medir (excepto fp8) está medido. §0.1 tiene los
resultados.

### Bloquea el PR de layout

1. **HND con `kv_cache_dtype=fp8`** `[?]` — explícitamente excluido del encargo.
   Cambia `padded_hs` y por tanto los strides; el 1.07× está medido solo en
   fp16/bf16. Es el único punto que impide fijar el default.

### Trabajo de integración (medido y especificado, no implementado)

2. **Guard de paso de fila** (§7.1): la condición es una línea en
   `triton_attn.py:383`, pero `kv_cache.split(hs, dim=-1)` (línea 687) asume que
   la última dim es exactamente `2*hs`. Hace falta el recorte de vista que ya usa
   la rama de cuantización per-token-head. **Ese es el trabajo real.**
3. **Portar el reparto de v6 (D∈{64,128,256}) al kernel paginado.** v6 es denso;
   el paginado es el que iría a vLLM.
4. **Aplicar la receta de VOPD por bancos** (§2.4) al kernel real. F8 no la
   probó: su kernel ya emite 74 `v_dual` sin tocarla.

### Abierto

5. **El 9.5 % hasta el roofline** (150.20 → 135.87 µs). No son bytes:
   `FETCH_SIZE` está a 1.03× del ideal. Es ancho de banda efectivo y latencia.
6. **Bancos de VGPR en WMMA**: 6 % teórico con receta verificada, pero solo
   aplica si `P@V` acabara en WMMA — y el veredicto es dot2 (§5.3).

### Descartado tras medir

- ~~Padding general del stride~~ — no transfiere; HND lo supera gratis. **Y
  padear mal (no múltiplo de 32 B) cuesta 1.47×.**
- ~~NSEG alto~~ — `NSEG=1` gana a D=256 (16 cuesta 700 µs); `NSEG=2` a D=128.
- ~~K/V non-temporal~~ — 1.84× **peor**: destruye el reuso de caché de GQA.
- ~~Q a LDS~~ — el kernel ganador usa 0 LDS.
- ~~`global_load_lds`~~ — no existe en gfx1151 (CDNA-only).
- ~~TILE/bloques grandes~~ — +52 % y +128 %.

## Fuentes

- `/scratch/rogarcia/rdna35-expert/` — wiki y microbenchmarks de RDNA3.5 con
  contadores HW. Especialmente `microbench/{wmma_cycles, wmma_bank_latency,
  wave32_vs_wave64_gemm, dot_cycles, vopd_promotion, lds_bank_stride,
  l2_dispatch_persistence, waitcnt_decrement_rate, cache_hierarchy}`.
- `raw/strix-ddr-bw-vs-wavefront.md`, `raw/fp16-gemm.md`, `raw/gemms-on-gfx11.md`
  — páginas de AMD MLSE Confluence (descargadas vía API REST).
- `raw/RDNA3_5_ISA.pdf` — referencia de la ISA.
- `vllm/v1/attention/ops/triton_unified_attention.py` y
  `vllm/v1/attention/backends/triton_attn.py`, rama
  `rogarcia.gfx1151-3d-attn-tuning`.
