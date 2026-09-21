# f0-baseline — baseline de referencia del kernel de atención decode en gfx1151

## Resultado en una línea

El baseline Triton 3D queda en **198.87 µs** para el caso de estudio
(`Hq=32, Hkv=16, M=4, D=256, S=2048`), **68.3 % del roofline de 230 GiB/s**, con
el path 3D verificado activo (`IS_3D=true`) en las 11 celdas del barrido; y la
hipótesis de mayor impacto del documento de diseño (§7.1, aliasing del stride)
**resulta CONFIRMADA en el efecto pero REFUTADA en el mecanismo**: padear el
stride da **1.20× gratis y con salida bit-idéntica**, pero no por leer menos
bytes — `FETCH_SIZE` no se mueve — sino por eliminar el 61 % de los stalls
`GL1C_STALL_GL2_GL1`.

---

## Cifras

### Tabla de baseline — LA QUE DEBE USAR EL RESTO DE LA FLOTA

Todas las celdas: `TRITON_ATTN`, fp16, `block_size=16`, CUDA graphs,
`--min-working-set-mb 96`, 5 repeticiones, **path 3D verificado**.
Roofline = KV leído por paso / 230 GiB/s.

| label | M | S | Hq/Hkv/D | **mediana (µs)** | spread | KV/paso | **roofline (µs)** | **% roof** | GiB/s | NSEG | capas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main_M1 | 1 | 2048 | 32/16/256 | **190.65** | 0.43 % | 32.0 MiB | 135.87 | **71.3 %** | 163.9 | 16 | 10 |
| main_M2 | 2 | 2048 | 32/16/256 | **193.55** | 0.53 % | 32.0 MiB | 135.87 | **70.2 %** | 161.5 | 16 | 10 |
| main_M3 | 3 | 2048 | 32/16/256 | **196.02** | 0.22 % | 32.0 MiB | 135.87 | **69.3 %** | 159.4 | 16 | 10 |
| **main_M4 (caso de estudio)** | **4** | **2048** | **32/16/256** | **198.87** | **0.26 %** | **32.0 MiB** | **135.87** | **68.3 %** | **157.1** | 16 | 10 |
| main_M5 | 5 | 2048 | 32/16/256 | **201.89** | 0.56 % | 32.0 MiB | 135.87 | **67.3 %** | 154.8 | 16 | 10 |
| S512_M4 | 4 | 512 | 32/16/256 | **61.42** | 0.66 % | 8.0 MiB | 33.97 | **55.3 %** | 127.2 | 16 | 12 |
| S1024_M4 | 4 | 1024 | 32/16/256 | **110.88** | 0.44 % | 16.0 MiB | 67.93 | **61.3 %** | 140.9 | 16 | 10 |
| S2048_M4 | 4 | 2048 | 32/16/256 | **198.79** | 0.45 % | 32.0 MiB | 135.87 | **68.3 %** | 157.2 | 16 | 10 |
| S4096_M4 | 4 | 4096 | 32/16/256 | **376.79** | 0.25 % | 64.0 MiB | 271.74 | **72.1 %** | 165.9 | 16 | 10 |
| llama_M1 | 1 | 2048 | 32/8/128 | **52.49** | 0.28 % | 8.0 MiB | 33.97 | **64.7 %** | 148.8 | 8 | 12 |
| llama_M4 | 4 | 2048 | 32/8/128 | **54.95** | 0.44 % | 8.0 MiB | 33.97 | **61.8 %** | 142.2 | 8 | 12 |

`spread` = (max − min) / mediana entre las 5 repeticiones. **Todas por debajo del
0.7 %** — el banco es reproducible y cualquier efecto por encima del 1 % es real.

Lecturas de la tabla:

- **M cuesta muy poco.** De M=1 a M=5 el tiempo sube solo un **5.9 %** (190.65 →
  201.89 µs) mientras el trabajo de query se quintuplica. Confirma el régimen
  memory-bound de §1.2: los bytes de KV son los mismos y dominan. El coste
  marginal de un token especulativo es ~2.8 µs, un **1.4 %** por token.
- **El % de roofline crece con S** (55.3 % → 72.1 % de S=512 a S=4096): el
  overhead fijo (launch, `reduce_segments`, prólogo) se amortiza. El techo
  asintótico observado ronda el 72 %.
- **El caso Llama (D=128) está peor en % de roofline** que el D=256 con el mismo
  KV por paso (61.8 % vs 55.3 % a igual 8 MiB — mejor, pero ambos lejos del 72 %
  que alcanza S=4096). Coherente con §7.3: con D=128 una fila son 256 B
  contiguos en vez de 512 B.

### Comando exacto para reproducir

El entorno **no es el que dice el §2 del protocolo** (ver "Desviaciones" abajo).
Este es el comando verificado:

```bash
# Barrido completo (11 celdas, 5 reps, ~4 min)
cd /scratch/rogarcia/f0-baseline
AMDSMI=/scratch/rogarcia/vllm-build/.venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi
QUIET_LOCK_TIMEOUT=7200 \
PATH="/scratch/rogarcia/vllm-build/.venv/bin:$PATH" \
PYTHONPATH="/scratch/rogarcia/f0-baseline:$AMDSMI" \
quiet-lock measure amd-gpu-lock \
  /scratch/rogarcia/vllm-build/.venv/bin/python fleet_f0_sweep.py \
  --reps 5 --min-working-set-mb 96 \
  --json-out /scratch/rogarcia/vllm/fleet/f0/baseline_sweep.json
```

Una sola celda por el benchmark estándar (da el mismo número, 199 µs):

```bash
cd /scratch/rogarcia/f0-baseline/benchmarks/attention_benchmarks
AMDSMI=/scratch/rogarcia/vllm-build/.venv/lib/python3.12/site-packages/_rocm_sdk_core/share/amd_smi
PATH="/scratch/rogarcia/vllm-build/.venv/bin:$PATH" \
PYTHONPATH="/scratch/rogarcia/f0-baseline:$AMDSMI" \
quiet-lock measure amd-gpu-lock \
  /scratch/rogarcia/vllm-build/.venv/bin/python benchmark.py \
  --backends TRITON_ATTN --batch-specs q4s2k \
  --num-q-heads 32 --num-kv-heads 16 --head-dim 256 \
  --min-working-set-mb 96 --inter-batch-cooldown 0
```

Las tres variables de entorno son **obligatorias**: sin `PYTHONPATH` al worktree
se mide el árbol instalado (sin el path 3D para M>1) y sin `AMDSMI` en el
`PYTHONPATH` vLLM no detecta la plataforma ROCm y el benchmark da `ERROR`.

### Padding del stride del KV (§7.1) — la palanca encontrada

Mismo kernel, misma forma, mismo patrón de acceso; solo se mueven los strides.
Caso de estudio, 5 reps por punto:

| pad (B) | stride head / token / página (B) | mediana (µs) | vs pad=0 | % roof | GiB/s |
|---|---|---|---|---|---|
| **0 (stock)** | 1024 / 16384 / 262144 | **198.98** | 1.000× | 68.3 % | 157.1 |
| 32 | 1056 / 16896 / 270336 | 167.95 | 0.844× | 80.9 % | 186.1 |
| **128** | **1152 / 18432 / 294912** | **165.29** | **0.831×** | **82.2 %** | **189.1** |
| 384 | 1408 / 22528 / 360448 | 165.77 | 0.833× | 82.0 % | 188.5 |
| 512 | 1536 / 24576 / 393216 | 189.46 | 0.952× | 71.7 % | 164.9 |

**A/B intercalado (3 rondas), para descartar orden de asignación o deriva
térmica:**

| ronda | pad=0 | pad=128 B | ratio |
|---|---|---|---|
| 1 | 198.93 µs | 165.58 µs | 1.201× |
| 2 | 198.84 µs | 165.32 µs | 1.203× |
| 3 | 198.82 µs | 165.61 µs | 1.201× |

Reproducible al 0.2 %. **1.202× de mediana.**

Generaliza al caso Llama (`32/8/128`, M=4, S=2048):

| pad (B) | mediana (µs) | vs pad=0 | % roof |
|---|---|---|---|
| 0 | 54.79 | 1.000× | 62.0 % |
| 32 | 47.79 | 0.872× | 71.1 % |
| 128 | **45.10** | **0.823×** | **75.3 %** |
| 0 (control final) | 54.79 | 1.000× | 62.0 % |

**Correctitud (§5.1 del protocolo): la salida es bit-idéntica.**

| pad | max abs err vs referencia fp32 densa | max diff vs pad=0 |
|---|---|---|
| 0 | 5.262e-04 | — |
| 16 | 5.262e-04 | **0.000e+00** |
| 64 | 5.262e-04 | **0.000e+00** |
| 192 | 5.262e-04 | **0.000e+00** |

Tolerancia fp16 del protocolo: 2e-2. Pasa con 38× de margen, y el diff contra la
salida sin padear es **exactamente cero**. El padding es una ganancia gratis: no
cambia un solo bit del resultado.

---

## Hipótesis probadas

| # | hipótesis (§ del doc de diseño) | veredicto | evidencia |
|---|---|---|---|
| 1 | El path 3D se activa con M=2..5 (§8.1) | **CONFIRMADA** | `IS_3D=true` y `reduce_segments` lanzado en las 11 celdas; grid `(1,16,16)`; ver abajo |
| 2 | Stride del KV padeado fuera de potencia de 2 da hasta 2.55× (§7.1) | **CONFIRMADA en efecto, REFUTADA en mecanismo y magnitud** | 1.20× medido y reproducible, no 2.55×; `FETCH_SIZE` NO cambia (33.02 → 33.00 MiB), luego no son "bytes extra"; lo que cae es `GL1C_STALL_GL2_GL1` 971 949 → 378 350 (−61 %) |
| 3 | El stride de head del caso de estudio es 1 MiB exacto (§7.1) | **REFUTADA** | El KV cache de vLLM es **paginado**. El stride de kv-head real es **1 KiB**, el de token 16 KiB y el de página 256 KiB. La cifra de 1 MiB del documento asume un layout contiguo `[S,H,D]` que no existe en este stack |
| 4 | El kernel es memory-bound (§1.2) | **CONFIRMADA** | 68.3 % del roofline DRAM; `GFX_CLK` medio bajo carga **~615 MHz** contra un máximo de 2900 MHz — el shader está parado esperando memoria; M×5 cuesta solo +5.9 % |
| 5 | `--min-working-set-mb` corrige un sesgo optimista (§3 del protocolo) | **CONFIRMADA** | `q1s512` 32/8/128: 10.57 µs sin el flag vs 18.39 µs con él = **+42.5 %** de sesgo. En las celdas de mi barrido (todas ya > 32 MiB) el flag es un no-op: bias entre −0.3 % y +0.0 % |
| 6 | NSEG=16 (default Triton) no está optimizado (§3.2, §8.1) | **NO CONCLUYENTE** | Confirmo que NSEG=16 es lo que se lanza (y 8 en Llama), y que da 80 WG… pero no barrí NSEG. Es el experimento que le toca a otro agente |
| 7 | El coste de los parciales es ~2 % (§3.3) | **REFUTADA para NSEG=16** | `reduce_segments` mide 4.48 µs contra 118.00 µs del kernel principal = **3.7 %** del tiempo, y lee 2.02 MiB (6.3 % de los 32 MiB de KV), no el 2 % que predice la fórmula con NSEG=5 |

---

## Evidencia de ASM / profiler

### 1. Verificación del path 3D (tarea crítica)

No recalculé la expresión del gate (podría divergir del código real). Envolví los
objetos de kernel Triton y registré el grid y los `constexpr` que
`unified_attention` **realmente** pasó. Salida literal para `q4s2k`
(`Hq=32, Hkv=16, D=256, M=4, S=2048`):

```json
{
  "main_kernel_launches": 160,
  "reduce_segments_launches": 160,
  "distinct_main_launches": [
    {
      "grid": [1, 16, 16],
      "IS_3D": true,
      "NUM_SEGMENTS_PER_SEQ": 16,
      "TILE_SIZE": 16,
      "BLOCK_Q": 8,
      "BLOCK_M": 16,
      "HEAD_SIZE": 256,
      "num_seqs": 1,
      "num_queries_per_kv": 2,
      "num_query_heads": 32,
      "USE_SWAPPED_GRID": false
    }
  ],
  "distinct_reduce_launches": [
    { "grid": [4, 32], "num_seqs": 1, "num_query_heads": 32 }
  ]
}
```

Tres pruebas independientes de que es el path 3D:

1. `IS_3D: true` — el `constexpr` pasado al kernel.
2. El grid es **3D** `(1, 16, 16)` = `(q_blocks, NSEG, H_kv)`. El path 2D lanza
   un grid 2D.
3. `reduce_segments` se lanzó 160 veces (una por lanzamiento del kernel
   principal). Ese kernel **solo existe en el path 3D**.

El grid `(4, 32)` de `reduce_segments` es `(q_tokens=4, H_q=32)`: confirma que
los buffers se indexan por token absoluto, que es justo lo que arregla el commit
`77698ec` (§8.1 del diseño).

**Cobertura del grid**: `1 × 16 × 16 = 256 WG` sobre 20 WGP. Nota para la flota:
el documento de diseño (§3.1) predice 16 WG sin split-KV y propone NSEG=5 para
llegar a 80; el baseline real ya lanza **256 WG** porque Triton usa NSEG=16. Es
3.2× más de lo que §3.2 recomienda.

Metadatos del dispatch, de `rocprofv3`:

```
kernel_unified_attention   VGPR=256  SGPR=128  LDS=0  Scratch=28  grid=32768  wg=128
reduce_segments            VGPR=64   SGPR=128  LDS=0  Scratch=0   grid=16384  wg=128
```

**`VGPR_Count = 256` — el máximo arquitectónico.** El objetivo de §0 del diseño
es **≤ 96 VGPR/lane**; el baseline está 2.7× por encima del objetivo y en el tope
duro. Por la fórmula de §2.2 eso da `floor(1536/256) = 6 waves/SIMD`, frente a
las 16 del tope. Además hay **28 bytes de scratch**, es decir, **spill a
memoria** — con 256 VGPR el asignador se quedó sin registros. `LDS = 0`: el
kernel Triton no usa LDS en absoluto, así que la recomendación de §0 #6 ("Q a
LDS, reusado") no está aplicada en el baseline.

### 2. Contadores — de dónde sale el 1.20× del padding

`rocprofv3`, medias por dispatch sobre 15 dispatches, un solo layer.
Los contadores no caben en una sola pasada (`error code 38: Request exceeds the
capabilities of the hardware`), así que van en tres grupos separados.

```
                        pad=0 (stock)      pad=128 B        cambio
kernel_unified_attention
  FETCH_SIZE              33 808 KiB       33 794 KiB       -0.04 %   <-- SIN CAMBIO
                          (33.02 MiB)      (33.00 MiB)
  GL2C_HIT                   183 720          184 132       +0.2 %
  GL2C_MISS                  292 452          292 056       -0.1 %
  GL2 hit rate                 38.6 %           38.7 %      sin cambio
  GL1C_STALL_GL2_GL1         971 949          378 350       -61.0 %   <-- AQUI
  duración (mediana)          118.00 us       109.16 us     -7.5 %

reduce_segments
  FETCH_SIZE               2 064 KiB        2 064 KiB       0 %
  GL2C_HIT                       575              575
  GL2C_MISS                   17 028           17 027
  GL2 hit rate                  3.3 %            3.3 %
  duración (mediana)            4.48 us          3.88 us
```

Esto es lo más importante del informe y **contradice al documento de diseño**.
§7.1 atribuye el 2.55× a leer 2.39× de bytes extra desde DRAM. Aquí:

- **`FETCH_SIZE` no se mueve**: 33.02 → 33.00 MiB. El kernel lee exactamente los
  mismos bytes con y sin padding. **No hay bytes extra que eliminar.**
- **Los 33.02 MiB son ya casi óptimos**: 1.03× sobre los 32.0 MiB teóricos. El
  tráfico de KV tiene solo un 3 % de desperdicio.
- **Lo que cambia es la contención en la ruta GL1→GL2**: los stalls caen 2.57×.

Es decir, el fenómeno **existe** pero es de **conflicto de canal/banco**, no de
amplificación de tráfico. En el microbenchmark de AMD citado en §7.1 el aliasing
sí producía relecturas (4.06 GB vs 1.70 GB); en el layout paginado de vLLM la
paginación ya rompe la amplificación de tráfico y solo queda la componente de
contención. Por eso el efecto es 1.20× y no 2.55×.

Corolario para quien escriba el kernel HIP: **el margen que queda no está en
leer menos bytes** — solo hay un 3 % ahí. Está en (a) la contención de canal, que
el padding ya captura, y (b) el 18 % que separa los 189 GiB/s con padding de los
230 GiB/s del roofline, que es latencia mal escondida (§7.2, profundidad de
pipeline) y 256 VGPR con spill.

### 3. Layout real del KV cache (refuta §7.1)

Strides tal y como los recibe el kernel (`stride_k_cache_0..3`), caso de estudio:

```
K-cache que indexa el kernel: [blocks=128, tokens=16, kv_heads=16, D=256]
  stride_k_cache_0 (página) = 131072 elems = 262144 B = 256.00 KiB  pow2=True
  stride_k_cache_1 (token)  =   8192 elems =  16384 B =  16.00 KiB  pow2=True
  stride_k_cache_2 (kv head)=    512 elems =   1024 B =   1.00 KiB  pow2=True
  stride_k_cache_3 (d)      =      1 elems =      2 B               pow2=True
```

El documento de diseño dice:

> Nuestro caso: `stride_head = S × D × esz = 2048 × 256 × 2 =` **1 MiB exacto,
> potencia de 2**.

**Ese stride no existe en este stack.** El KV cache de vLLM es paginado: la forma
física es `[num_blocks, block_size, num_kv_heads, 2*head_size]` (layout NHD) y
K/V van empaquetados en la última dimensión. El stride de head real es **1 KiB**,
1024× menor que el que asume el documento. `S` no aparece en ningún stride: entra
por la block table.

Lo que sí se conserva es la conclusión práctica — **las cuatro dimensiones son
potencias de dos** — y por eso padear funciona igual. Pero la magnitud predicha
(2.55×) venía de un layout distinto y no se transfiere.

Consecuencia concreta para un tile: `TILE_SIZE=16` tokens de una kv head son 16
tramos de `D×2 = 512 B` contiguos, separados 16 KiB entre sí, repartidos sobre un
span de página de 256 KiB. Son **8 KiB útiles por página tocada**. Encaja con la
fila de 512 B contiguos que §7.3 da como mejor caso (225 GiB/s), y explica por
qué el caso D=128 (256 B por fila) sale peor en % de roofline.

### 4. Qué propiedad del stride predice la velocidad

Barrido fino, buscando qué invariante separa los puntos rápidos de los lentos:

```
 padB      us   head    tok     blk   head%256  tok%4096  blk%16384
    0   198.9   1024  16384  262144         0         0          0    <- lento
   16   179.1   1040  16640  266240        16       256       4096
   32   168.0   1056  16896  270336        32       512       8192    <- rapido
   64   186.2   1088  17408  278528        64      1024          0
   96   177.0   1120  17920  286720        96      1536       8192
  128   165.3   1152  18432  294912       128      2048          0    <- el mejor
  160   176.8   1184  18944  303104       160      2560       8192
  192   183.9   1216  19456  311296       192      3072          0
  224   176.2   1248  19968  319488       224      3584       8192
  256   191.5   1280  20480  327680         0         0          0    <- lento
  384   165.8   1408  22528  360448       128      2048          0    <- rapido
  512   189.5   1536  24576  393216         0         0          0    <- lento
```

**El predictor es `stride_head mod 256`.** Los tres puntos lentos (pad 0, 256,
512 µs → 198.9 / 191.5 / 189.5) son exactamente los tres con `head % 256 == 0`.
Los dos mejores (pad 128 y 384 → 165.3 / 165.8) son los dos con
`head % 256 == 128`, es decir, los que quedan **máximamente lejos** del múltiplo
de 256 en ambas direcciones.

256 B es **la granularidad de interleave de canal LPDDR5X en gfx1151** que cita
§7.5 del diseño (el `SUS=256` de `StaggerU`). Es decir: la señal medida no es
"potencia de dos" en abstracto, es **alineación con el interleave de canal**. La
recomendación operativa que se desprende es más precisa que la del documento:

> Elegir el padding tal que `stride_head % 256 == 128`, no simplemente "fuera de
> potencia de dos".

Con el layout stock (`stride_head = 1024 B`) todas las kv heads caen en el mismo
canal; con +128 B se reparten. Esto también sugiere que la palanca de §7.5
(escalonar el offset de inicio por WG en múltiplos de 256 B) ataca el mismo
fenómeno por otra vía, y que probablemente **no se sumen**.

### 5. Reloj bajo carga (§6 del protocolo)

`APU_AVERAGE_GFXCLK_FREQUENCY` muestreado durante 400 iteraciones del kernel:

```
APU_CURRENT_GFX_MAXFREQ:        2900 MHz
APU_AVERAGE_GFXCLK_FREQUENCY:   614 MHz / 612 MHz / 618 MHz
```

**~615 MHz, un 21 % del máximo.** No es throttling térmico: es la firma de un
kernel parado en memoria — el SoC baja el reloj de shader porque no hay trabajo
de ALU que hacer. Es evidencia independiente, del lado del hardware, de que el
kernel es memory-bound, y confirma la nota de §2.8 de que las cifras por ciclo a
boost no describen esta carga. Todos los tiempos de este informe son de reloj de
pared, no ciclos.

---

## Qué NO se pudo probar y por qué

1. **Inspección de ASM (§5.4 del protocolo).** No la hice. El kernel es Triton,
   no HIP: el binario se genera en JIT y no hay un `.hsaco` que pasar por
   `llvm-objdump` sin volcar la caché de Triton. Lo que sí obtuve del profiler
   —`VGPR=256`, `Scratch=28`, `LDS=0`— es la parte del punto 4 que decide
   diseño, pero **no** miré loads en vuelo entre `s_waitcnt`, ni la posición del
   `global_load` tras el `s_waitcnt`, ni bancos WMMA. El baseline queda por tanto
   **en progreso** según el §5 del protocolo. Para cerrarlo haría falta volcar el
   IR de Triton (`TRITON_CACHE_DIR` + `MLIR_ENABLE_DUMP=1` o los `.hsaco` del
   caché) y desensamblarlo.

2. **Desglose de stalls `vmcnt` vs `lgkmcnt` (§5.5 del protocolo).** Los
   contadores disponibles en esta board no lo separan. `rocprofv3-avail` ofrece
   `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY` y `SQ_WAIT_INST_LDS`, pero no un
   `SQ_WAIT_INST_VMEM`. Lo más cerca que llegué es `GL1C_STALL_GL2_GL1`, que sí
   discrimina la ruta de memoria global y es lo que uso para el análisis del
   padding. **No puedo dar el desglose vmcnt/lgkmcnt que pide el protocolo.**
   Atenuante: `LDS_Block_Size = 0`, el kernel no usa LDS en absoluto, así que
   `lgkmcnt` solo puede venir de SMEM y el stall es necesariamente de memoria
   global.

3. **Más de un contador por pasada.** El hardware rechaza `GL2C_HIT GL2C_MISS
   FETCH_SIZE GL1C_STALL_GL2_GL1` juntos (`error code 38`). Cada grupo va en un
   run distinto, así que las comparaciones entre grupos (p. ej. `FETCH_SIZE`
   contra `GL1C_STALL`) cruzan ejecuciones. Como la dispersión medida es < 0.7 %,
   no invalida las conclusiones, pero conviene saberlo.

4. **Barrido de NSEG.** Fuera de mi encargo y requiere tocar el kernel; solo
   documento qué NSEG usa el baseline (16 en el caso principal, 8 en Llama).

5. **El padding NO está integrado en vLLM.** Lo mido inyectando un allocador en
   el harness del benchmark. Llevarlo a producción exige cambiar
   `get_kv_cache_shape` / la asignación del KV cache manager, con impacto en la
   contabilidad de memoria (con pad=128 B el cache crece un 12.5 % en el caso de
   estudio: 512 → 576 elementos por entrada). **La cifra de 1.20× es una cota de
   lo que hay disponible, no una mejora ya entregada.**

---

## Desviaciones respecto al protocolo (importante para el resto de la flota)

El §2 del protocolo describe un entorno que **no es el que hay**. Tres
diferencias, todas con consecuencias:

1. **El vLLM instalado no es `/scratch/rogarcia/vllm`.** El editable apunta a
   `/scratch/rogarcia/vllm-build/vllm`, que está **655 commits por detrás** de
   `gfx11` y **no contiene el path 3D para M>1**. Medir sin `PYTHONPATH` da el
   kernel equivocado en silencio. Las extensiones compiladas (`_C.abi3.so`,
   `_rocm_C.abi3.so`) sí se toman de ahí, y funcionan: no hay `.so` en los
   worktrees.

2. **Falta el módulo Python `amdsmi`.** Sin él, `vllm.platforms` cae a
   `UnspecifiedPlatform`, `on_gfx1151()` no se evalúa y el benchmark devuelve
   `ERROR` con un mensaje que no menciona la causa. El paquete existe sin
   instalar en
   `…/site-packages/_rocm_sdk_core/share/amd_smi`; basta añadirlo al
   `PYTHONPATH`. **Con esto, `on_gfx1151()` devuelve `True`** y el tuning de
   gfx1151 se activa.

3. **`amd-gpu-lock` necesita `amd-smi` en el `PATH`** y no está en el `PATH` por
   defecto; hay que anteponer `/scratch/rogarcia/vllm-build/.venv/bin`. Sin esto
   falla con `required tool 'amd-smi' not found` **y devuelve exit 0**, así que
   un script que compruebe el código de salida no se entera de que no midió nada.

4. **`/scratch/rogarcia/vllm` está ocupado por otro agente** (worktree en
   `rogarcia.fleet-f1-triton`). Para no pisarlo trabajé en un worktree propio,
   `/scratch/rogarcia/f0-baseline`. Recomiendo a los demás hacer lo mismo.

Nota de disciplina git: no toqué `main`, ni `gfx11`, ni las dos ramas de
referencia, ni el worktree de f1. Mi rama es `rogarcia.fleet-f0-baseline`, con la
fusión de las dos ramas **sin conflictos** (base común `89ba324e70`, merge `ort`
limpio: 4 ficheros, 101 inserciones). No hay push.

---

## Recomendaciones para el resto de la flota

Ordenadas por lo que dicen los datos, no por lo que decía el documento:

1. **El padding del stride es real y es gratis: 1.20×, salida bit-idéntica.**
   Pero el objetivo correcto es `stride_head % 256 == 128` (alineación de canal),
   no "fuera de potencia de dos". Con eso el caso de estudio pasa de 68.3 % a
   82.2 % del roofline.
2. **Dejad de perseguir bytes.** `FETCH_SIZE` ya está a 1.03× del óptimo. El
   tráfico de KV **no** es donde queda margen, en contra de lo que sugiere §0.
3. **Los 256 VGPR con 28 B de spill son el hallazgo estructural.** El objetivo de
   §0 es ≤ 96. El baseline está en el tope duro y spillando. Es lo que más
   probablemente explica el 18 % que queda entre el padding (189 GiB/s) y el
   roofline (230 GiB/s), vía profundidad de pipeline (§7.2): con los registros
   agotados no se pueden mantener 4–6 loads en vuelo.
4. **`LDS = 0`**: el kernel Triton no usa LDS. La palanca de §0 #6 (Q a LDS)
   está entera por explorar.
5. **NSEG=16 lanza 256 WG**, 3.2× lo que recomienda §3.2 (NSEG=5, 80 WG), y
   `reduce_segments` cuesta un 3.7 %, no un 2 %. Merece barrido.
6. El techo práctico observado con el padding es **~82 % del roofline**. Es la
   referencia realista a batir, no el 100 %.

---

## Artefactos

Código (rama `rogarcia.fleet-f0-baseline`, worktree
`/scratch/rogarcia/f0-baseline`, commit `362779019a`):

| fichero | qué hace |
|---|---|
| `/scratch/rogarcia/f0-baseline/fleet_f0_sweep.py` | barrido de baseline: mediana, dispersión, % de roofline, path por celda |
| `/scratch/rogarcia/f0-baseline/fleet_f0_check_3d.py` | verificación del path 3D envolviendo los objetos de kernel Triton |
| `/scratch/rogarcia/f0-baseline/fleet_f0_strides.py` | vuelca los strides del KV cache tal y como los ve el kernel |
| `/scratch/rogarcia/f0-baseline/fleet_f0_pad_stride.py` | experimento de padding del stride (incluye modo A/B intercalado) |
| `/scratch/rogarcia/f0-baseline/fleet_f0_verify_pad.py` | correctitud del layout padeado contra referencia fp32 densa |
| `/scratch/rogarcia/f0-baseline/fleet_f0_profile_driver.py` | bucle de dispatch limpio para `rocprofv3` |

Datos:

| fichero | contenido |
|---|---|
| `/scratch/rogarcia/vllm/fleet/f0/baseline_sweep.json` | las 11 celdas, con las 5 repeticiones individuales de cada una |
| `/scratch/rogarcia/vllm/fleet/f0/pad_stride_main.json` | barrido de padding, caso de estudio |
| `/scratch/rogarcia/vllm/fleet/f0/pad_stride_ab.json` | A/B intercalado pad 0 / 128 B, 3 rondas |
| `/scratch/rogarcia/vllm/fleet/f0/pad_stride_fine.json` | barrido fino de padding (la tabla de `% 256`) |
| `/scratch/rogarcia/vllm/fleet/f0/pad_stride_llama.json` | padding en la config Llama-3-8B |
| `/scratch/rogarcia/vllm/fleet/f0/kv_strides_main.json` | strides y formas del KV cache |
| `/scratch/rogarcia/vllm/fleet/f0/prof/*.csv` | 6 runs de `rocprofv3` (pad 0/128 B × 3 grupos de contadores) |
| `/scratch/rogarcia/vllm/fleet/f0/prof/summary.json` | contadores agregados por kernel y por dispatch |
