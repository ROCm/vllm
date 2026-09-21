# F3b — optimización del camino de memoria del kernel HIP

Estado: **OBJETIVO CUMPLIDO**. Kernel final: `fleet/hip/decode_attn_v5.hip`.

## Resultado

| variante | µs | vs partida | % roofline (135.87 µs) |
|---|---|---|---|
| Roofline (230 GiB/s, 32 MiB KV) | 135.87 | — | 100 % |
| Baseline Triton (a batir) | 198.87 | — | 68.3 % |
| Triton con NSEG ajustado | 185.0 | — | 73.5 % |
| **Partida (f3a `decode_attn.hip`)** | **445.07** | 1.00× | 30.5 % |
| v5 `NSEG=1 KPW=4 KV_PAD=0` | 166.96 | 2.67× | **81.4 %** |
| **v5 `NSEG=1 KPW=4 KV_PAD=64`** | **155.55** | **2.86×** | **87.3 %** |

Desglose del mejor caso: **`decode_attn` 154.85 µs + `reduce_segments` 2.86 µs**
(200.9 GiB/s de KV sobre un techo de 230). Repetido 3× con ITERS=200: 154.76 /
154.97 / 155.06 µs — desviación < 0.2 %.

**Bate el baseline de 198.87 µs por un 21.8 %** y queda a 19.7 µs del roofline.

## Qué se hizo

### Optimización 1 — rediseño del reparto por lane (2.67×, 445 → 167 µs)

Era lo que pedía el brief y es donde está casi todo el beneficio. `acc` pasa de
`[M]` (una dim de salida por lane) a `[M][8]`: el lane `l` de una wave posee las
dims `8l..8l+7`, así que **una fila K o V de 256 fp16 es exactamente un
`global_load_b128` por lane** y 512 B contiguos por wave.

Consecuencias en el ASM (`fleet/hip/attn_v5.s`, medido con `fleet/hip/asmstat.sh`):

| | f3a (partida) | v5 |
|---|---|---|
| ancho de carga | `global_load_d16_b16` (2 B/lane) | **12 × `global_load_b128`** (16 B/lane) |
| loads en vuelo | 0 | **9** |
| `s_barrier` | 7 por tile | **0 en todo el kernel** |
| LDS | 4352 B | **0** |
| producto punto | 52 `v_fma_mix_f32`, 0 `v_dot2` | 128 `v_fma_mix`, 18 `v_dot2acc_f32_f16`, 83 `v_dual` |
| VGPR / spill | 81 / 0 | **129 / 0** |

Los 129 VGPR concuerdan exactamente con el dato de F2 para fdot2 (129 limpio).
La eliminación de barriers sale gratis del diseño: cada wave es un flujo de
online-softmax independiente sobre su propio subconjunto de keys, y los 8 flujos
se fusionan escribiendo cada uno su propio segmento a global (v4) en vez de por
LDS. `reduce_segments` ya hacía esa fusión; solo hay que darle `NSEG*NWAVE`
segmentos en lugar de `NSEG`.

### Optimización 2 — NSEG=1 (1.10×, ~185 → 167 µs)

**Es el segundo parámetro en importancia y va al revés de lo esperado.** Con el
kernel nuevo, menos segmentos es más rápido, monótonamente:

| NSEG | µs (attn solo) |
|---|---|
| 1 | **165.6** |
| 2 | 167.0 |
| 4 | 260.7 |
| 8 | 324.3 |
| 16 | 700.1 |

Con NSEG=1 ya hay 8 waves × 32 bloques = 256 waves de paralelismo, suficiente
para 40 CU; subir NSEG solo multiplica el conjunto de trabajo vivo. **El coste
no está en el epílogo ni en `reduce_segments`** (2.9 µs a NSEG=1, 14.4 µs a
NSEG=8 — nunca la diferencia de 160 µs): está dentro del bucle de atención, es
decir, es localidad de caché, no sincronización.

### Optimización 3 — `KV_PAD=64` (1.07×, 167 → 155.6 µs)

Padding de 64 elems fp16 = 128 B por fila, `stride_head` 512 → 640 B.
**Repetible y bit-idéntico en salida.** El barrido es no monótono, típico de
conflictos de sets de caché:

| KV_PAD (elems) | stride_head (B) | µs |
|---|---|---|
| 0 | 512 | 166.9 |
| 8 | 528 | 173.1 |
| 32 | 576 | 177.1 |
| 48 | 608 | 193.6 |
| **64** | **640** | **155.8** |
| 96 | 704 | 178.9 |
| 128 | 768 | 182.7 |
| 192 | 896 | 156.2 |

Nota para el manager: esto **no contradice** lo que midió F2. F2 midió el layout
NHD real de vLLM con `stride_head = 1 KiB` y ahí el statu quo ya es óptimo. Mi
harness usa layout denso propio con `stride_head = 512 B`, un punto distinto de
la curva. La regla `% 256 == 128` del brief efectivamente **no se sostiene**:
mi óptimo (640 B) da `640 % 256 = 128` pero 768 B también lo cumple y es de los
peores del barrido. **Es un efecto de caché específico del stride concreto; hay
que medirlo, no derivarlo de una regla.** Como este padding no existe en el
layout de vLLM, la cifra que conviene citar como resultado transferible es la de
`KV_PAD=0`: **166.96 µs**, que ya bate el baseline por un 16 %.

## Lo que NO funcionó (medido, no supuesto)

1. **Non-temporal en K/V (`NT=1`): 1.84× PEOR** (319.5 vs 173.9 µs). La receta
   R6 de `f4-asm.md` no aplica a este kernel. El motivo es que con GQA=2 dos
   bloques de q-head leen la misma cabeza KV, así que se emiten 64 MiB para
   32 MiB únicos y **la caché recupera la mitad del tráfico**. Marcar
   non-temporal destruye precisamente esa reutilización.
2. **Fusión GQA (v3, `fleet/hip/decode_attn_v3.hip`): neutra o peor**
   (183.4 vs 174.1 µs). Un bloque por cabeza KV en vez de por cabeza Q elimina
   la amplificación 2× de raíz — pero también reduce los bloques de 32 a 16 y
   dobla los registros de query (`NROW = MAXM*GQA = 8` filas). El paralelismo
   perdido cuesta más que el tráfico ahorrado. **Correcto** (max_rel 1.7e-5),
   conservado por si sirve con GQA mayor.
3. **Interleaving de segmentos sobre el grid (v5 `ILV=1`): neutro**
   (165.6 vs 167.2 µs, dentro del ruido). Hacer que todas las waves recorran el
   KV en lockstep no arregla el escalado con NSEG. Es el default porque no hace
   daño, pero no es una optimización.
4. **`s_setprio`: neutro** (173.91 vs 174.11 µs, 0.1 %). Con 12 `b128` ya en
   vuelo no hay burst que proteger. Coste cero, sin beneficio; queda tras
   `-DSETPRIO=1`, apagado.
5. **Más profundidad de pipeline: peor.** KPW=2 → 176.0, KPW=4 → **167.2**,
   KPW=6 → 201.5, KPW=8 → 174.6 µs. Concuerda con el dato de F2 de que satura
   en 4 loads en vuelo. Aquí KPW=4 son 8 cargas `b128` emitidas por iteración.
6. **Bloques mayores: peor.** BLOCK=256 → 167.2, 512 → 188.2, 1024 → 267.4 µs.
   Con 129 VGPR la ocupación por SIMD ya está limitada por registros.

## Correctitud — los tres criterios del brief, verificados

`fleet/hip/final.sh` ejecuta **20 casos**: M ∈ {1,2,3,4,5} × S ∈ {2048, 48} ×
KV_PAD ∈ {0, 64}. **20/20 PASS**, `max_rel` entre 1.65e-05 y 3.95e-05, tres
órdenes de magnitud por debajo del criterio de 1e-3.

**Control negativo**: `-DMUTATE=1` introduce un off-by-one en la máscara causal.

| | max_abs | max_rel | veredicto |
|---|---|---|---|
| criterio viejo (`max_abs ≤ 2e-2`) a S=2048 | 1.249e-03 | — | **habría PASADO** |
| criterio nuevo (`max_rel ≤ 1e-3`) a S=2048 | — | 1.065e+00 | FAIL |
| criterio nuevo a S=48 | 5.440e-02 | 3.732e+01 | FAIL |

**Confirmado el fallo del protocolo que señalaba el brief**: el bug real habría
pasado con `max_abs`. `max_rel` lo detecta por un factor de 1000×, y a S=48 el
error ni siquiera se diluye (`max_abs` también lo pilla). El validador nuevo
está en `fleet/hip/check.py` (`max_rel` como criterio, S=48 obligatorio).

## Artefactos

Todo en `/scratch/rogarcia/vllm/fleet/hip/` (copia de trabajo en
`/scratch/rogarcia/f3-hip/fleet/hip/`). No se tocó nada fuera de `fleet/`, no se
commiteó nada, no se usó `git clean`.

| ruta | qué es |
|---|---|
| `decode_attn_v5.hip` | **kernel final** — b128, 0 barriers, 0 LDS, interleaving opcional |
| `decode_attn_v4.hip` | v5 sin interleaving (epílogo wave→global) |
| `decode_attn_v3.hip` | variante con fusión GQA (correcta, no más rápida) |
| `decode_attn_v2.hip` | primer rediseño, epílogo por LDS (histórico) |
| `check.py` | validador `max_rel ≤ 1e-3` |
| `final.sh` | validación completa: 20 casos + control negativo |
| `sweep.sh` | barrido de variantes, un solo lock para N builds y N medidas |
| `asmstat.sh` | mezcla de instrucciones, loads en vuelo, VGPR, spill |
| `attn_v5.s` | ASM gfx1151 del kernel final |

### Reproducir

```bash
source /scratch/rogarcia/vllm/fleet/fleetenv.sh /scratch/rogarcia/f3-hip
cd /scratch/rogarcia/f3-hip/fleet/hip
bash final.sh                                    # 20 casos + control negativo + tiempo
SRC=$PWD/decode_attn_v5.hip bash sweep.sh "-DNSEG=1 -DKPW=4 -DKV_PAD=64"
bash asmstat.sh /tmp/v5.s
```

## Limitaciones

1. Solo `D=256, HQ=32, HKV=16`, fp16. El `static_assert(DPL == 8)` fija el
   diseño a 8 fp16 por lane; con D=128 y wave32 saldría `b64` y haría falta
   repartir 2 lanes por fila.
2. Layout denso propio, no el paginado de vLLM. `KV_PAD=64` es un artefacto de
   ese layout — **no transferir esa cifra**; la transferible es 166.96 µs.
3. Una sola secuencia, batch=1.
4. El techo de 230 GiB/s se hereda de la flota; no lo remedí de forma
   independiente. A 200.9 GiB/s queda ~13 % de margen y el kernel es ya casi
   puramente memory-bound (0 barriers, 0 LDS, sin spill).
