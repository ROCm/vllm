# F8 — Acantilado de stride (T1) y D=128 en el kernel HIP (T2)

## Resumen en una línea

**T1: el acantilado es real y afecta a Gemma4 (D=512), pero no es lo que decía
el brief.** El predictor correcto **no** es `stride_head`, es el **paso de fila**
(`row_pitch = 2*head_size*esz`, la extensión de la última dimensión en bytes),
que es `stride_head` bajo NHD y `stride_token` bajo HND. La pérdida medida
end-to-end en vLLM es **1.09× bajo NHD y 1.32× bajo HND** para D=512 — no 1.90×.
Y el padding mínimo que la cura **no es de 256 B**: `+32 B` captura el 82 % de la
ganancia por **1.6 %** de memoria en vez de 12.5 %. El guard se justifica, con
tres avisos.

**T2: el kernel HIP generalizado a D=128 da 40.7 µs contra los 54.95 µs de
Triton (1.35×, 81 % del roofline), sin romper D=256** (168.60 vs 167.35 µs) y
con **128 VGPR y cero spills**. `fleet/hip/decode_attn_v6.hip`.

---

## T1 — ¿es real el acantilado en vLLM?

### 1. Qué `stride_head` produce realmente `get_kv_cache_shape`

`get_kv_cache_shape` devuelve la forma lógica
`(num_blocks, num_kv_heads, block_size, 2*head_size)`
(`vllm/v1/attention/backends/triton_attn.py:356`). K y V van **empaquetados** en
la última dimensión, así que el paso de fila es `2 * head_size * esz`.

La confusión del brief: qué stride lleva ese paso **depende del layout**.

| layout | `stride_order` | `stride_head` | `stride_token` |
|---|---|---|---|
| NHD (por defecto) | `(0,2,1,3)` | `2·hs·esz` (**el paso de fila**) | `Hkv · 2·hs·esz` |
| HND | `(0,1,2,3)` | `bs · 2·hs·esz` | `2·hs·esz` (**el paso de fila**) |

Bajo HND el `stride_head` es `block_size` veces mayor y es **siempre** múltiplo
de 2048 para cualquier `hs ≥ 64` — si el predictor fuera `stride_head % 2048`,
HND estaría *siempre* en el acantilado, y F6 mide que HND es **más rápido**. El
predictor tiene que ser el paso de fila. Lo confirman las medidas de abajo.

Pasos de fila que produce el backend hoy (`Hkv` y `block_size` no intervienen):

| head_size | fp16 | fp8 |
|---:|---:|---:|
| 64 | 256 | 128 |
| 128 (Llama-3, Qwen, Mistral) | 512 | 256 |
| 192 | 768 | 384 |
| 256 | 1024 | 512 |
| **512 (Gemma4 global)** | **2048 ← acantilado** | 1024 |
| 576 (MLA) | 2304 | 1152 |
| **1024** | **4096 ← acantilado** | **2048 ← acantilado** |

### 2. ¿Cae alguna config real en la trampa? → **Sí: Gemma4.**

`vllm/transformers_utils/configs/gemma4.py:27` — Gemma4 usa `global_head_dim`
en sus capas `full_attention`, y los tests del árbol lo fijan en **512**
(`tests/v1/attention/test_mm_prefix.py:505`, «512: Gemma4 global head dim»;
`tests/v1/core/test_contiguous_kv_packing.py:30`). En fp16 eso es exactamente
`row_pitch = 2048 B`. **Es un modelo en árbol, no una hipótesis.**

DeepSeek V3/V4 con `head_dim=512` va por MLA, que tiene su propio backend y
`get_kv_cache_shape`; no pasa por aquí. El caso que importa es Gemma4.

### 3. Medida end-to-end en vLLM (no microbenchmark)

Banco: `TRITON_ATTN`, fp16, `block_size=16`, CUDA graphs, `Hq=32, Hkv=16, M=4,
S=2048`, 3 reps intercaladas round-robin (así la deriva térmica pega a los dos
brazos por igual). Harness: `fleet/f8_pad_stride_e2e.py`. Se re-aloja el KV cache
con `pad` elementos extra en la dim de contenido y se recorta la vista: **misma
forma, mismo patrón de acceso, solo se mueven los strides**. `pad=0` es la
asignación stock y hace de control.

**D=512 bajo NHD (el layout por defecto hoy):**

| pad | row_pitch | %2048 | µs | GiB/s | vs pad0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 2048 | **0** | 1163.55 | 53.7 | 1.00× |
| +256 B | 2304 | 256 | 1077.84 | 58.0 | **1.09×** |
| +512 B | 2560 | 512 | 1028.10 | 60.8 | **1.13×** |
| +1024 B | 3072 | 1024 | 1039.49 | 60.1 | 1.12× |
| +2048 B | 4096 | **0** | 1530.91 | 40.8 | **0.76× (peor)** |

**D=512 bajo HND (lo que F6 recomienda):**

| pad | row_pitch | %2048 | µs | GiB/s | vs pad0 |
|---:|---:|---:|---:|---:|---:|
| 0 | 2048 | **0** | 1155.23 | 54.1 | 1.00× |
| +64 B | 2112 | 64 | 969.29 | 64.5 | 1.19× |
| +128 B | 2176 | 128 | 896.78 | 69.7 | 1.29× |
| +256 B | 2304 | 256 | 874.98 | 71.4 | **1.32×** |
| +512 B | 2560 | 512 | 877.90 | 71.2 | 1.32× |
| 0 (control repetido) | 2048 | 0 | 1155.23 | 54.1 | 1.000× |

El control cierra exacto: sin deriva.

**Conclusión: 1.09× bajo NHD, 1.32× bajo HND. No 1.90×.** El 1.90× del
microbenchmark de F2 no se transfiere: allí el acantilado se producía padeando
un `D=256` hasta un stride de 2048 B, es decir **con un agujero de 1 KiB por
fila**. Con `D=512` el paso de 2048 B está **lleno**, y eso cambia tanto el
patrón de sets como la eficiencia de las cargas. El brief pedía comprobar
exactamente esto y la respuesta es que el microbenchmark sobreestimaba 1.4–1.7×.

### 4. El predictor es el paso de fila, y es `%2048 == 0`. Control negativo limpio.

Control decisivo: **si el `+256 B` ayudase por cualquier motivo genérico,
ayudaría también donde el paso no es múltiplo de 2048.** No lo hace — ahí
*perjudica*:

| D | row_pitch base | %2048 | efecto de `+256 B` |
|---:|---:|---:|---|
| 128 | 512 | 512 | **1.10× peor** (116.09 → 127.23 µs) |
| 192 | 768 | 768 | 1.01× peor (306.38 → 309.20 µs) |
| 256 | 1024 | 1024 | **1.24× peor** (281.65 → 348.26 µs) |
| **512** | **2048** | **0** | **1.09× mejor** (1163.55 → 1077.84) |
| **1024** | **4096** | **0** | **1.06× mejor** (1994.19 → 1879.16) |

Padear ayuda **solo y exactamente** en los dos pasos múltiplos de 2048.
En los demás cuesta hasta un 24 %. Esto cierra el argumento: el guard debe
dispararse por condición, **nunca aplicarse en general**.

Y con agujero el acantilado sí es grande: barrido a D=256 bajo NHD padeando
hasta pitch 4096 — pitch 3840 → 348.40 µs, **pitch 4096 → 586.59 µs**, pitch
4352 → 348.63 µs. **1.68× de caída con vecinos sanos a los dos lados**,
exactamente donde F2 lo predecía. El fenómeno existe; lo que no se transfiere es
su magnitud cuando el paso está lleno en vez de agujereado.

### 5. Dependencia de H_kv (lo que preguntaba el manager) → **el guard debe condicionarse a H_kv**

D=512, NHD, `+256 B`, barriendo `Hkv`:

| Hkv | pad0 µs | pad+256B µs | ganancia |
|---:|---:|---:|---:|
| 1 | 737.54 | 734.26 | **1.004× — nada** |
| 2 | 625.02 | 607.26 | 1.029× |
| 4 | 638.06 | 612.46 | 1.042× |
| 8 | 694.68 | 702.27 | **0.99× — nada** |
| 16 | 1163.55 | 1077.84 | **1.09×** |

Con `Hkv ≤ 8` el efecto está en el ruido o es negativo; **solo con `Hkv = 16`
paga**. Esto reproduce de forma independiente el mecanismo de conflicto de sets
que F2 estableció (hacen falta ≥ 2 columnas vivas, y satura al llenar el set) y
coincide con la escala de H_kv que F6 midió para HND (Hkv=4 → 0.99×, Hkv=8 →
1.022×, Hkv=16 → 1.073×). **El guard no debe pagar memoria donde no hay
ganancia.**

### 6. El padding mínimo: `+32 B`, no `+256 B`

Barrido fino bajo HND, D=512, Hkv=16:

| pad | row_pitch | µs | ganancia | coste de memoria |
|---:|---:|---:|---:|---:|
| 0 | 2048 | 1167.48 | 1.00× | — |
| **+8 B** | 2056 | 1711.57 | **0.68× ¡MUCHO peor!** | +0.4 % |
| **+16 B** | 2064 | 1708.56 | **0.68× ¡MUCHO peor!** | +0.8 % |
| **+32 B** | 2080 | 955.69 | **1.22×** | **+1.6 %** |
| **+48 B** | 2096 | 1702.43 | **0.69× ¡MUCHO peor!** | +2.3 % |
| +64 B | 2112 | 965.78 | 1.21× | +3.1 % |
| +96 B | 2144 | 909.21 | 1.28× | +4.7 % |
| +128 B | 2176 | 894.99 | 1.30× | +6.2 % |
| +256 B | 2304 | 873.37 | **1.34×** | **+12.5 %** |

Dos cosas, ambas importantes:

1. **`+32 B` captura el 82 % de la ganancia (1.22× de 1.34×) por 1.6 % de
   memoria en vez de 12.5 %.** Ocho veces menos coste por cuatro quintos del
   beneficio. Es el punto que hay que elegir.
2. **Los pads de 8, 16 y 48 B son un 47 % PEORES que no padear.** No son ruido:
   spread 0.3–1.05 %, y 32 y 64 B (a ambos lados de 48) salen bien. Es la
   alineación: 32 B es el ancho de un `dwordx8`, y 2080 = 65·32 mantiene la fila
   alineada a 32 B mientras 2096 = 65.5·32 no. **Cualquier guard que elija el
   padding con aritmética ingenua puede aterrizar en 48 B y empeorar las cosas
   un 47 %.** El padding tiene que ser múltiplo de 32 B.

---

## T1 — El guard propuesto

### Dónde

`vllm/v1/attention/backends/triton_attn.py`, en el `return` final de
`TritonAttentionBackend.get_kv_cache_shape` (línea 383 hoy):

```python
return (num_blocks, num_kv_heads, block_size, 2 * head_size)
```

### Qué condición

```python
# A row of packed K|V is 2*head_size*esz bytes.  When that pitch is a
# multiple of 2 KiB every kv-head column maps to the same cache set and
# the GL1<-GL2 path stalls (1.09x under NHD, 1.32x under HND, measured
# on gfx1151 at head_size=512).  One 32-byte step off the multiple cures
# most of it; 8/16/48-byte steps are *worse* than not padding, so the pad
# must keep the row 32-byte aligned.  Only >=16 kv-heads contend enough
# for this to pay, so narrower models keep the tight layout.
pitch = 2 * head_size * get_dtype_size(cache_dtype)
if pitch % 2048 == 0 and num_kv_heads >= 16:
    pad = 32 // get_dtype_size(cache_dtype)
    return (num_blocks, num_kv_heads, block_size, 2 * head_size + pad)
```

Con la salvedad de que el backend tiene que recortar la vista a `2*head_size`
antes de entregarla al kernel (como ya hace `_pth_key_value_caches` con el
padding de escalas per-token-head, `triton_attn.py:783`): la ruta de
`kv_cache.split(hs, dim=-1)` de la línea 687 asume que la última dim es
exactamente `2*hs`, así que el guard necesita el mismo tratamiento de vista que
la rama de cuantización. **Ese es el trabajo real de implementación**, no la
condición.

### Qué cuesta

`page_size_bytes` crece un **1.6 %** (2048 → 2080 B por fila), luego el número
de bloques del KV cache cae un 1.6 %. Solo para configs con
`row_pitch % 2048 == 0 && Hkv >= 16`, que hoy es **Gemma4 en fp16** y
`head_size=1024` (fp16 o fp8). Todo lo demás —Llama-3, Qwen, Mistral (D=128),
D=256, MLA D=576— no toca el guard y no paga un byte.

### Recomendación honesta

**Ganancia 1.22× por 1.6 % de memoria en el caso afectado: se justifica.** Pero
con tres avisos que el brief no anticipaba:

1. **El número es 1.22×–1.32×, no 1.90×.** El 1.90× del microbenchmark medía un
   paso *agujereado*; el caso real tiene el paso lleno.
2. **El guard debe condicionarse a `Hkv >= 16`.** Con Hkv ≤ 8 no hay ganancia
   que justifique el coste (medido: 1.004× a Hkv=1, 0.99× a Hkv=8).
3. **El padding debe ser múltiplo de 32 B.** 8, 16 y 48 B son un 47 % peores que
   no padear. Un guard que "desalinee un poco" sin cuidar los 32 B es
   activamente dañino.

Sobre la pregunta del manager (¿sigue valiendo bajo HND?): **sí, y rinde más**.
HND **no** arregla el acantilado — bajo HND el caso D=512 sigue en 1155 µs y el
padding sigue dando 1.32× frente al 1.09× de NHD. Son ortogonales: HND ataca el
`stride_head` y el guard ataca el paso de fila. **La conclusión no cambia entre
layouts y la recomendación es la misma en los dos.**

### Artefactos T1

| fichero | contenido |
|---|---|
| `/scratch/rogarcia/vllm/fleet/f8_pad_stride_e2e.py` | harness end-to-end, pads round-robin, reporta row_pitch |
| `/scratch/rogarcia/vllm/fleet/f8/d512_sweep.json` | D=512 NHD, barrido 2048→4096 |
| `/scratch/rogarcia/vllm/fleet/f8/d512_hnd_fine.json` | D=512 HND, con control repetido |
| `/scratch/rogarcia/vllm/fleet/f8/d512_hnd_knee2.json` | el knee del padding (la tabla de 8/16/32/48 B) |
| `/scratch/rogarcia/vllm/fleet/f8/d512_hkv{1,2,4,8}.json` | dependencia de H_kv |
| `/scratch/rogarcia/vllm/fleet/f8/D{128,192,256}_p128.json` | controles negativos |
| `/scratch/rogarcia/vllm/fleet/f8/d256_holesweep.json`, `d256_4096nbrs.json` | acantilado con agujero (1.68× a pitch 4096) |
| `/scratch/rogarcia/vllm/fleet/f8/d1024.json`, `d1024_hnd.json` | segundo punto del acantilado |

---

## T2 — D=128 en el kernel HIP → **funciona, 1.35× sobre Triton**

### Resultado

**40.7 µs** para `Hq=32, Hkv=8, D=128, M=4, S=2048` contra los **54.95 µs** del
baseline Triton que midió F0: **1.35×**, y **187 GiB/s** (81 % del roofline de
230 GiB/s, frente al 61.8 % de Triton). Kernel: `fleet/hip/decode_attn_v6.hip`,
config `NSEG=2 KPW=4`.

Y **D=256 no se rompe**: 168.60 µs con v6 contra 167.35 µs con v5, un 0.7 % de
diferencia con 128 VGPR en vez de 129. La generalización no cuesta nada en el
caso que v5 ya servía.

### Cómo: opción B, pero sin el "dos filas por lane"

El brief planteaba A (dos lanes por fila, `b64`) y B (dos filas por lane,
`b128`). **Elegí la B pero con un giro que le quita la complicación**: en vez de
hacer que un lane cargue dos tokens, **mantengo `DPL = 8` fijo (siempre `b128`)
y parto la wave**:

```c
#define DPL 8                  // fp16 per lane = one global_load_b128, always
#define LPR (HEAD_DIM / DPL)   // lanes that together cover one K/V row
#define SUB (WAVE / LPR)       // independent token streams per wave
```

- D=256 → `LPR=32, SUB=1`: una fila por wave. **Exactamente el reparto de v5.**
- D=128 → `LPR=16, SUB=2`: dos grupos de 16 lanes, **cada uno en su propio
  token**. Sigue siendo un `b128` por lane; lo que cambia es que la wave tiene
  dos tokens en vuelo en vez de uno.
- D=64 → `LPR=8, SUB=4`. Validado también.

La clave que hace esto barato: **los SUB streams nunca tienen que encontrarse.**
Cada grupo de lanes ya es un stream de softmax online independiente, así que
simplemente **se convierte en su propio segmento de salida** y
`reduce_segments` funde `NSEG*NWAVE*SUB` en vez de `NSEG*NWAVE`. Eso evita de
raíz lo que el brief temía de la opción B ("complica el reparto y la máscara
causal"):

- **La máscara causal no se complica**: cada lane sabe su token
  (`jj = jb + c*SUB + sub`), así que la condición se queda idéntica.
- **La reducción del dot product** pasa de barrer los 32 lanes a barrer solo los
  `LPR` que comparten fila (`for (st = 1; st < LPR; st <<= 1)`). `LPR` es
  potencia de dos, así que la mariposa xor nunca se sale del grupo.
- **Cero barreras y cero LDS**, igual que v5.

Los `SUB` streams se intercalan (`c*SUB + sub`) para que los grupos de una wave
toquen tokens adyacentes en el mismo ciclo en vez de separados por `SUB*KPW`.

### Tuning

| config | µs | GiB/s | vs Triton (54.95) |
|---|---:|---:|---:|
| KPW=2, NSEG=1 | 58.31 | 134.0 | 0.94× |
| KPW=4, NSEG=1 | 45.24 | 172.7 | 1.21× |
| KPW=6, NSEG=1 | 63.62 | 122.8 | 0.86× |
| KPW=8, NSEG=1 | 47.14 | 165.7 | 1.17× |
| **KPW=4, NSEG=2** | **40.7** | **187.4** | **1.35×** |
| KPW=4, NSEG=4 | 76.09 | 102.7 | 0.72× |
| KPW=4, NSEG=8 | 142.11 | 55.0 | 0.39× |

Confirmación intercalada, 3 rondas: NSEG=1 → 47.42/47.10/46.59, NSEG=2 →
40.82/40.64/40.63 µs. Separación limpia, sin solape.

**NSEG=2 es lo contrario de lo que v5 quería a D=256** (allí NSEG=1 ganaba y
subirlo empeoraba monótonamente). Tiene sentido: con `Hkv=8` y D=128 el grid es
la mitad de ancho, así que a NSEG=1 falta paralelismo para llenar las 40 CU; a
NSEG=4 ya vuelve a dominar el conjunto de trabajo vivo. El óptimo se ha movido,
no la física.

### Registros (el punto que levantó F7)

| kernel | VGPR | spill |
|---|---:|---:|
| v5 D=256 | 129 | 0 |
| v6 D=256 | 128 | 0 |
| **v6 D=128 NSEG=2** | **128** | **0** |

**128 VGPR y cero spills**, muy lejos del techo de 256. Es el margen que F7
decía que hace falta: con la pendiente de dot2 (+0.9 VGPR por operando de V
residente) hay sitio de sobra para subir operandos; con WMMA (+6) no lo habría.
El mix confirma que el camino es el correcto: **12 `global_load_b128`** (la
carga ancha se mantuvo, que era el objetivo de la tarea), 128 `v_fmac_mix`,
**74 `v_dual`** ya emparejados, 0 `ds_*`, 0 `s_barrier`, y 9 cargas en vuelo
como máximo.

Nota para F7: no he ido a por la receta de VOPD de los bancos de VGPR. El
kernel ya sale a 74 `v_dual` sin tocarla, y con 128 VGPR libres es la palanca
obvia si alguien quiere seguir. La intuición del manager de que la opción B daría
más operandos P distintos es correcta —con `SUB=2` hay dos streams con P
distinta— pero no lo he medido por separado.

### Correctitud

Criterio del brief: `max_rel ≤ 1e-3` + caso corto + control negativo. Todo
verde, y **el control negativo es el que justifica el criterio**:

| caso | max_rel | veredicto |
|---|---:|---|
| D=128 M=4 S=2048 | 1.852e-05 | PASS |
| D=128 M=4 S=48 | 2.194e-05 | PASS |
| D=128 M=1 S=2048 / S=48 | 2.09e-05 / 2.34e-05 | PASS |
| D=128 M=5 S=2048 / S=48 | 1.65e-05 / 3.38e-05 | PASS |
| D=128 NSEG=2 S=2048 / S=48 | 1.368e-05 / 2.194e-05 | PASS |
| D=256 M=4 S=2048 / S=48 (regresión) | 1.895e-05 / 3.431e-05 | PASS |
| D=64 M=4 S=2048 / S=48 | 2.145e-05 / 2.665e-05 | PASS |
| **MUTANTE (off-by-one causal) S=2048** | **9.521e-01** | **FAIL ✓** |
| **MUTANTE (off-by-one causal) S=48** | **3.400e+01** | **FAIL ✓** |

El mutante a S=2048 da `max_abs = 1.218e-03`, que **habría pasado un criterio de
`max_abs ≤ 2e-2` sin despeinarse**. Confirma lo que avisaba el brief: el
criterio correcto es `max_rel`, y el caso corto (S=48) lo hace 35× más
llamativo.

### Artefactos T2

| fichero | contenido |
|---|---|
| `/scratch/rogarcia/vllm/fleet/hip/decode_attn_v6.hip` | el kernel generalizado (D ∈ {64,128,256}) |
| `/scratch/rogarcia/vllm/fleet/hip/out/dev_d128_n2.s` | asm device-only de D=128 NSEG=2 |

Reproducir:

```bash
cd /scratch/rogarcia/vllm/fleet/hip
source /scratch/rogarcia/vllm/fleet/fleetenv.sh /scratch/rogarcia/vllm
build "$HIPCC" -O3 --offload-arch=gfx1151 -o out/v6 \
    -DHEAD_DIM=128 -DNUM_KV_HEADS=8 -DMAXM=4 -DNSEG=2 -DKPW=4 decode_attn_v6.hip
HQ=32 HKV=8 D=128 "$PY" check.py gen 2048 4 104 out/d
measure ./out/v6 2048 200 out/d/q.bin out/d/k.bin out/d/v.bin out/d/got.bin
HQ=32 HKV=8 D=128 "$PY" check.py check out/d out/d/got.bin 4
```

---

## Qué queda abierto

- **T1**: implementar el guard de verdad requiere el recorte de vista en
  `kv_cache.split` — no lo he hecho, solo he medido y propuesto. Tampoco he
  medido el guard con un modelo Gemma4 real cargado (el banco usa formas
  sintéticas con las mismas dimensiones).
- **T2**: no he probado la receta de VOPD por bancos de F7 sobre v6, ni he
  portado el reparto de v6 al kernel **paginado** (`decode_attn_paged.hip`), que
  es el que tendría que ir a vLLM. v6 es el denso.
